"""Regression tests for the public search and service surface."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INQUIRY_URL = (
    "https://kim3310-doeon-kim-portfolio.pages.dev/"
    "?offer=tool-call-finetune-lab&inquiry=agent-reliability-audit#private-inquiry"
)


def test_service_offer_uses_central_agent_reliability_audit_lane() -> None:
    for path in [ROOT / "docs/service-offer.json", ROOT / "site/service-offer.json"]:
        offer = json.loads(path.read_text())

        assert offer["lead_capture_url"] == INQUIRY_URL
        assert offer["commerce"]["lane_id"] == "agent-reliability-audit"
        assert offer["commerce"]["lane_name"] == "Agent Reliability Audit"

        paid_offer = offer["structured_data"]["offers"][1]
        assert paid_offer["name"] == "fixed-scope Agent Reliability Audit"
        assert paid_offer["url"] == INQUIRY_URL


def test_public_site_states_synthetic_demo_boundary_without_benchmark_claims() -> None:
    html = (ROOT / "site/index.html").read_text()

    for expected in [
        "Request private audit",
        "Try synthetic demo",
        "fixed-scope Agent Reliability Audit",
        "credential-free and synthetic",
        INQUIRY_URL,
    ]:
        assert expected in html

    for disallowed in [
        "96.4% valid calls",
        "paid dataset preparation pack",
        "View paid options",
    ]:
        assert disallowed not in html


def test_search_growth_notes_use_central_private_inquiry_route() -> None:
    notes = (ROOT / "docs/search-growth-implementation.md").read_text(encoding="utf-8")

    assert INQUIRY_URL in notes
    assert "central private inquiry route" in notes
    assert "offer=tool-call-finetune-lab" in notes
    assert "inquiry=agent-reliability-audit" in notes
    assert "GitHub Issue Form" not in notes
