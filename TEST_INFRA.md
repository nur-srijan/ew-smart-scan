# E2E Test Infra: EW SmartScan Engine & C2-ESM TOC Dashboard

## Test Philosophy
- Opaque-box, requirement-driven. Derived strictly from ORIGINAL_REQUEST.md.
- Verification mechanism is independent of internal implementation details.
- 4-Tier Methodology:
  - Tier 1: Feature Coverage (≥5 test cases per feature covering happy paths and nominal behavior)
  - Tier 2: Boundary & Corner Cases (≥5 test cases per feature covering limits, extremes, noise, and edge conditions)
  - Tier 3: Cross-Feature Combinations (Pairwise interaction coverage across engine, timing, telemetry, and dashboard)
  - Tier 4: Real-World Application Scenarios (Realistic multi-node electronic warfare missions with mixed radar environments)

## Feature Inventory Mapping
| # | Feature | Source | Tier 1 | Tier 2 | Tier 3 |
|---|---------|--------|:------:|:------:|:------:|
| 1 | C++20 RMAB Closed-Form Index | ORIGINAL_REQUEST §13 | 5 | 5 | ✓ |
| 2 | C++20 Bayesian Belief Updating | ORIGINAL_REQUEST §14 | 5 | 5 | ✓ |
| 3 | Multi-Tuner Top-M Selection | ORIGINAL_REQUEST §13,49 | 5 | 5 | ✓ |
| 4 | CMake & pybind11 Build System | ORIGINAL_REQUEST §15,41 | 5 | 5 | ✓ |
| 5 | High-Res Latency Benchmark (<100ns) | ORIGINAL_REQUEST §16,42 | 5 | 5 | ✓ |
| 6 | Drop-in Python Wrappers | ORIGINAL_REQUEST §17 | 5 | 5 | ✓ |
| 7 | 50µs Dwell Timing Simulation | ORIGINAL_REQUEST §20-22 | 5 | 5 | ✓ |
| 8 | Timing Jitter Statistics Export (<5µs) | ORIGINAL_REQUEST §23,45 | 5 | 5 | ✓ |
| 9 | Fleet Telemetry Matrix | ORIGINAL_REQUEST §27,49 | 5 | 5 | ✓ |
| 10 | Interactive Multi-Tuner Waterfall | ORIGINAL_REQUEST §28,51 | 5 | 5 | ✓ |
| 11 | EOB Threat Identification Table | ORIGINAL_REQUEST §29,50 | 5 | 5 | ✓ |
| 12 | PDW Intercept Log & Data Export | ORIGINAL_REQUEST §30,50 | 5 | 5 | ✓ |
| 13 | AI vs Legacy Comparison HUD | ORIGINAL_REQUEST §31 | 5 | 5 | ✓ |
| 14 | Mathematical Equivalence Tests | ORIGINAL_REQUEST §34,44 | 5 | 5 | ✓ |
| 15 | Environment Integration Tests | ORIGINAL_REQUEST §35 | 5 | 5 | ✓ |
| 16 | Headless Dashboard Verification | ORIGINAL_REQUEST §36,48 | 5 | 5 | ✓ |
| 17 | Full Regression Integrity (87 tests) | ORIGINAL_REQUEST §54 | 5 | 5 | ✓ |

## Test Architecture
- Test Runner: `uv run pytest tests/e2e -v`
- Pass/Fail Semantics: 100% assertions pass, zero exceptions, zero timeouts, zero deadline violations.
- Directory Layout:
  - `tests/e2e/test_tier1_features.py`
  - `tests/e2e/test_tier2_boundaries.py`
  - `tests/e2e/test_tier3_pairwise.py`
  - `tests/e2e/test_tier4_scenarios.py`

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| S1 | Multi-UAV Cooperative Air Defense Patrol | F1, F2, F3, F6, F9, F10, F15 | High |
| S2 | High-Density FHSS Agile Threat Interception | F1, F2, F3, F5, F7, F8, F11, F12 | Extreme |
| S3 | Synchronized Fleet Zero-Collision Surveillance | F3, F9, F10, F13, F15 | High |
| S4 | Mixed Fixed & Scanning Emitter EOB Mapping | F2, F11, F12, F15 | High |
| S5 | Rapid Electronic Dwell Budget Stress under Load | F1, F5, F7, F8, F14 | Extreme |

## Coverage Thresholds
- Tier 1: ≥85 test cases (≥5 across 17 features)
- Tier 2: ≥85 test cases (≥5 across 17 features)
- Tier 3: ≥17 cross-feature interaction test cases
- Tier 4: ≥5 realistic operational scenarios
- Total: ≥192 test cases
