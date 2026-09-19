# Orchestration Plan: EW SmartScan Engine & C2-ESM TOC Dashboard

## Objectives
Deliver a production-ready, zero-allocation C++20 RMAB Whittle Index engine with pybind11 bindings (<100ns latency) and 50µs hardware timing harness, alongside a redesigned modern C2-ESM Tactical Operations Center (TOC) Dashboard with multi-payload fleet telemetry, live RF waterfall, EOB threat library, and PDW export.

## Phase 0: Survey (Parallel Exploration)
- Dispatch 3 Explorers:
  - Explorer 1: Focus on `ew_sim` Python RMAB Whittle Index implementation, mathematical equations, Bayesian belief state update formulas, and existing `WhittleIndexScheduler` / `MultiWhittleScheduler` classes.
  - Explorer 2: Focus on project build system, C++20 compiler capabilities, pybind11 configuration, uv environment, test setup (`uv run pytest`), and benchmark methodology.
  - Explorer 3: Focus on `demo/dashboard.py`, `DynamicSpectrumEnv`, `MultiReceiverSpectrumEnv`, pulse/emitter data structures, PDW schema, fleet telemetry models, and UI requirements.

## Phase 1: Scope & Decomposition
- Synthesize Survey reports into `PROJECT.md`:
  - Feature Inventory (mapping every feature from R1-R4)
  - Milestone decomposition (M1 to M4)
  - Code Layout and Interface Contracts
- Establish `TEST_INFRA.md` for requirement-driven E2E testing track.

## Phase 2: Dual Track Execution
- Implementation Track:
  - M1: Zero-Allocation C++20 RMAB Engine & pybind11 Extension (K=35, <100ns latency, zero heap allocation on hot path, drop-in Python wrapper)
  - M2: 50µs Hardware Dwell Timing Loop Simulation (<5µs total jitter, monotonic clock timing, statistics export)
  - M3: Military-grade C2-ESM TOC Dashboard (`demo/dashboard.py` redesign: fleet telemetry matrix, 2D RF waterfall, EOB threat library, PDW export, AI vs legacy comparison HUD)
  - M4: Automated Verification & Regression Suite (numerical equivalence, env integration, headless UI rendering, 87/87 existing tests passing)
- E2E Testing & Hardening Track:
  - Comprehensive Tiers 1-4 tests covering all features and boundaries.
  - Tier 5 adversarial verification and forensic auditing (100% clean).

## Phase 3: Final Acceptance & Sentinel Reporting
- Gate verification across all modules.
- Present final synthesized results to Sentinel with complete verification evidence.
