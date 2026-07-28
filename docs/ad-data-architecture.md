# Ad-Supported Resource and Aggregate Data Architecture

Repository: `tool-call-finetune-lab`

## Public Resource Model

Free tool-call dataset boundary checklist for fine-tune preparation and eval reviews.

- Audience: LLM evaluation engineers and dataset reviewers
- Central resource: https://kim3310-doeon-kim-portfolio.pages.dev/resources/tool-call-finetune-lab/
- Live system: https://tool-call-finetune-lab.pages.dev/
- Advertising boundary: ads allowed only on public dataset-boundary resources; raw examples, prompts, eval outputs, and dashboards are ad-free
- Current ad state: code-ready on the central resource; serving depends on Google AdSense site approval and consent policy.

## Readiness Utility

The central resource turns the repository architecture into a practical review checklist:

- **Architecture Summary:** Repository-local proof surface for agent runtime reliability and AI workflow orchestration, backed by Python service or lab runtime, Container build surface, Local compose environment.
- **Runtime And Data Flow:** Primary domain: agent runtime reliability and AI workflow orchestration.
- **Cloud Or Local Deployment Boundary:** Operating model: stateless runtimes, provider adapters, queue-aware execution, telemetry, and controlled secret boundaries
- **Deployment patterns:** Containerized runtime path suitable for repeatable local, staging, or managed service deployment Stateless agent gateway with provider abstraction, retries, cost controls, and trace capture
- **Control boundaries:** identity boundary and least-privilege service access environment separation for local, staging, and managed runtime paths secret storage outside source and deterministic fallback for missing credentials observability hooks for logs, metrics, traces, and audit events rollback path...

The checklist state remains in the visitor's browser and is not transmitted.

## Aggregate Data Boundary

- Data asset: anonymous aggregate tool-call dataset-control interest and checklist usage counts
- Sensitivity class: agent-reliability-public
- Allowed events: `resource_view`, `resource_cta_click`, `architecture_doc_open`, `privacy_support_open`
- Prohibited fields: `raw_input`, `prompt`, `url`, `referrer`, `title`, `user_id`, `session_id`, `ip_address`, `payment_detail`
- Consent defaults to off.
- DNT and Global Privacy Control fail closed.
- Events are reduced to repository, allowlisted event, public surface, and consent-policy version.
- Personal, sensitive, raw, event-level, or re-identifiable data is never offered for sale.

## Storage Path

```text
Public resource
  -> consent and privacy-signal gate
  -> Cloudflare Pages event API
  -> rate-limited daily aggregate counter
  -> public benchmark response
  -> Firebase public aggregate data mart
```

Cloudflare D1 holds operational counters. Firestore project `kim3310-free-tools` is the deny-by-default public aggregate data mart. Private inquiries remain isolated from telemetry.
