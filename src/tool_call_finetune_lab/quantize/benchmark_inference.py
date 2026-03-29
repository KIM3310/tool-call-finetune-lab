"""Inference benchmarking: measure latency (p50/p95/p99) and throughput.

Runs against a live vLLM endpoint and reports token-level statistics.

Usage:
    python -m tool_call_finetune_lab.quantize.benchmark_inference \
        --url http://localhost:8000/v1 \
        --model qwen2.5-7b-tool-call \
        --concurrency 1 4 8 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default test prompts covering different tool-call complexities
BENCHMARK_PROMPTS = [
    {
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    },
    {
        "messages": [{"role": "user", "content": "Search for Python documentation."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Book a flight from NYC to LA for next Monday, returning Friday.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_flights",
                    "description": "Search for flights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string"},
                            "destination": {"type": "string"},
                            "departure_date": {"type": "string"},
                            "return_date": {"type": "string"},
                        },
                        "required": ["origin", "destination", "departure_date"],
                    },
                },
            }
        ],
    },
]


async def _single_request(
    client: Any,
    model: str,
    prompt: Dict[str, Any],
    max_tokens: int = 256,
) -> Tuple[float, int, bool]:
    """Send one request and return (latency_s, output_tokens, success)."""
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()

    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=prompt["messages"],
                tools=prompt.get("tools", []),
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=0.0,
            ),
        )
        latency = time.perf_counter() - t0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return latency, output_tokens, True
    except Exception as e:
        latency = time.perf_counter() - t0
        logger.warning("Request failed: %s", e)
        return latency, 0, False


async def run_concurrent_benchmark(
    client: Any,
    model: str,
    concurrency: int,
    n_requests: int,
    max_tokens: int = 256,
) -> Dict[str, Any]:
    """Run n_requests with a given concurrency level.

    Returns a stats dict with latency percentiles and throughput.
    """
    prompts = BENCHMARK_PROMPTS * (n_requests // len(BENCHMARK_PROMPTS) + 1)
    prompts = prompts[:n_requests]

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(prompt: Dict[str, Any]) -> Tuple[float, int, bool]:
        async with semaphore:
            return await _single_request(client, model, prompt, max_tokens)

    logger.info("Running %d requests with concurrency=%d...", n_requests, concurrency)
    wall_start = time.perf_counter()
    tasks = [bounded_request(p) for p in prompts]
    raw_results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - wall_start

    latencies = [r[0] for r in raw_results if r[2]]
    output_tokens = [r[1] for r in raw_results if r[2]]
    n_success = sum(1 for r in raw_results if r[2])
    n_failed = n_requests - n_success

    if not latencies:
        return {
            "concurrency": concurrency,
            "n_requests": n_requests,
            "n_success": 0,
            "n_failed": n_requests,
            "error": "All requests failed",
        }

    latencies_sorted = sorted(latencies)
    total_tokens = sum(output_tokens)

    def percentile(data: List[float], p: float) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "n_success": n_success,
        "n_failed": n_failed,
        "wall_time_s": round(wall_time, 3),
        "requests_per_sec": round(n_success / wall_time, 2),
        "latency_p50_s": round(percentile(latencies_sorted, 50), 3),
        "latency_p95_s": round(percentile(latencies_sorted, 95), 3),
        "latency_p99_s": round(percentile(latencies_sorted, 99), 3),
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "latency_min_s": round(min(latencies), 3),
        "latency_max_s": round(max(latencies), 3),
        "total_output_tokens": total_tokens,
        "tokens_per_sec": round(total_tokens / wall_time, 1),
        "avg_output_tokens": round(statistics.mean(output_tokens), 1) if output_tokens else 0,
    }


def run_benchmark(
    base_url: str,
    model: str,
    concurrency_levels: List[int],
    n_requests: int,
    max_tokens: int,
    output_file: str,
) -> List[Dict[str, Any]]:
    """Run benchmarks at multiple concurrency levels and save results."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    # Warmup
    logger.info("Warming up with 3 requests...")
    for _ in range(3):
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("Warmup request failed: %s", e)

    all_results: List[Dict[str, Any]] = []

    for c in concurrency_levels:
        result = asyncio.run(
            run_concurrent_benchmark(
                client=client,
                model=model,
                concurrency=c,
                n_requests=n_requests,
                max_tokens=max_tokens,
            )
        )
        all_results.append(result)
        _print_result(result)

    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Benchmark results saved to %s", output_file)

    return all_results


def _print_result(result: Dict[str, Any]) -> None:
    """Pretty-print a single benchmark result."""
    print(f"\n--- Concurrency: {result['concurrency']} ---")
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  Requests:        {result['n_success']}/{result['n_requests']} succeeded")
    print(f"  Wall time:       {result['wall_time_s']:.3f}s")
    print(f"  Requests/sec:    {result['requests_per_sec']:.2f}")
    print(f"  Tokens/sec:      {result['tokens_per_sec']:.1f}")
    print(f"  Latency p50:     {result['latency_p50_s'] * 1000:.0f}ms")
    print(f"  Latency p95:     {result['latency_p95_s'] * 1000:.0f}ms")
    print(f"  Latency p99:     {result['latency_p99_s'] * 1000:.0f}ms")
    print(f"  Avg output tok:  {result['avg_output_tokens']:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM inference benchmarking")
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen2.5-7b-tool-call")
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16],
        help="Concurrency levels to test",
    )
    parser.add_argument("--n-requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="results/inference_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        base_url=args.url,
        model=args.model,
        concurrency_levels=args.concurrency,
        n_requests=args.n_requests,
        max_tokens=args.max_tokens,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
