# Service Architecture - tool-call-finetune-lab

This document defines the deployment and resource plan for turning this repository into a buyer-reviewable service. It intentionally avoids public financial assumptions, public financial guesses, or contract assumptions.

## Commercial Role

- **Lane:** Tool-call evaluation and adaptation
- **Primary buyer:** AI teams improving tool-call accuracy and test coverage
- **First motion:** Model adaptation study using approved datasets and reproducible evals

## Recommended Architecture

Static docs/report surface, Python eval runner in CI or container, optional storage for datasets and scorecards.

~~~text
Visitor or operator
  -> public proof surface
  -> scoped app/API layer when a buyer workflow needs state
  -> managed data, object storage, queue, and observability only after scope is approved
  -> signed report, demo, export, or operating handoff
~~~

## Resource Plan

| Resource | Use | Buy timing |
| --- | --- | --- |
| Static hosting | GitHub Pages or Cloudflare Pages for a public, cacheable proof surface with custom-domain routing later. | Already sufficient for proof surfaces unless a custom domain is needed. |
| App/API runtime | Render, Fly.io, Railway, or Cloudflare Workers for a small API runtime only after a real workflow needs server state. | Buy only when a pilot needs authenticated workflows, integrations, or server-side jobs. |
| Data layer | Supabase or Neon Postgres for relational state; Cloudflare D1 only when the app is Workers-first and relational needs are small. | Buy after the workflow has real state, roles, or audit history. |
| Object storage | Cloudflare R2 or S3-compatible storage for uploads, reports, screenshots, model artifacts, or signed exports. | Buy when reports, uploads, signed exports, or model artifacts must persist. |
| Queue/cache | Upstash Redis/QStash or Cloudflare Queues for async jobs, retries, scheduled checks, and rate-limited workflows. | Buy when jobs, retries, scheduling, rate limits, or async processing appear. |
| Observability | Sentry plus privacy-safe web analytics for errors, performance, and buyer-flow learning without storing private visitor data. | Enable before external users test the service. |

## Repo-Specific Resources

- GitHub Pages
- container runner
- artifact storage
- OpenAI/fine-tuning provider account if needed

## Information Needed From Account Owner

- provider API key
- dataset storage credentials

## Revenue Channel Architecture

- **Primary channel:** B2B diagnostics and pilots
- **Monetization route:** Paid readiness review, scoped proof-of-value pilot, then implementation or governance support.
- **Acquisition motion:** Portfolio proof, GitHub evidence, direct outreach to AI, data, platform, and security leaders.
- **Activation path:** Confirm workflow owner, approved sample data, success criteria, and minimum cloud resources before connecting real integrations.
- **Margin control:** Keep the pilot fixed-scope, reuse templates and evidence packs, set spend limits, and add persistent infrastructure only after buyer scope is approved.
- **Public boundary:** Keep revenue, pricing, and contract assumptions in private planning; this repository publishes service structure, resource needs, and technical proof only.

## Cost and Risk Controls

- dataset license review
- batch/offline processing for cost control
- human review of scorecards

## Production Readiness Checklist

- Public demo route or README proof link is current.
- Service boundary states what the system does and does not do.
- Data storage, retention, and deletion path are defined before private data is accepted.
- Secrets are stored in platform secret managers, never committed to the repo.
- Spend limits, usage alerts, or manual approval gates are enabled before buyer testing.
- Logs and analytics avoid private payloads.
- Rollback or disable path exists for every external integration.
