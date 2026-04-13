# AGENTS.md — AgriBiomasse

## Scope & style
- Prefer **minimal diffs** and local fixes.
- **No broad refactor** unless explicitly requested.
- Preserve current **Streamlit UX** (layout/flow/text) unless a change is necessary to fix a bug.
- Keep solutions simple; avoid hidden complexity and over-engineering.

## Stability first (Streamlit)
- Prioritize reliability and memory safety when handling large datasets.
- Avoid loading/rending more data than needed; keep map/data operations bounded.
- Keep API integrations optional and fault-tolerant (graceful fallback to local data when possible).

## Execution approach
- For non-trivial tasks, propose a short **plan-first** approach before editing.
- Implement incrementally (small safe patches), then validate.
- If behavior changes, update `README.md` in the same change.
- After each implementation, always provide **manual test steps**.

## Local validation (known commands)
- Commands clearly documented in this repo:
  - `pip install -r requirements.txt`
  - `streamlit run app.py`
- If extra checks are used (e.g., syntax checks), state explicitly that they are auxiliary and not documented project test commands.
- If a validation command is uncertain/unavailable in the repo, say so explicitly.

## Safe incremental roadmap (V1 / V2 / V3)
- **V1 (stabilize):** bug fixes, guardrails, memory-safe defaults, no UX disruption.
- **V2 (optimize):** targeted performance improvements with parity checks.
- **V3 (extend):** optional features (e.g., API enrichments) behind safe fallbacks and without breaking existing flows.
