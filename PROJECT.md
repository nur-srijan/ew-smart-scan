# Project: EW SmartScan C++20 RMAB Engine & C2-ESM Tactical Operations Center

## Architecture
The system consists of three primary functional layers:
1. **Zero-Allocation C++20 Core (`rmab_cpp`)**:
   - High-performance, zero-heap-allocation Restless Multi-Armed Bandit (RMAB) Whittle Index scheduler for $K=35$ sub-bands.
   - Closed-form Whittle index calculations and Bayesian belief updates executed with zero dynamic memory allocation on hot execution paths.
   - Built via CMake and pybind11, exporting C++ classes `RMABSchedulerCore` and `MultiRMABSchedulerCore` with Python wrapper compatibility.
   - Hardware timing engine simulating 50µs slot dwell transitions with jitter profiling using `std::chrono::steady_clock`.
2. **Simulation & Environment Layer (`ew_sim`)**:
   - Spectrum simulation environments (`DynamicSpectrumEnv`, `MultiReceiverSpectrumEnv`) modeling fixed, agile frequency hopping (FHSS), and scanning radar emitters.
   - Seamless drop-in integration allowing either native Python schedulers or compiled C++ `rmab_cpp` schedulers with 100% numerical parity.
3. **C2-ESM Tactical Operations Center Dashboard (`demo/dashboard.py`)**:
   - Modern military-grade C2-ESM UI implemented in Plotly Dash with Flask WSGI backend.
   - Distributed fleet telemetry matrix across 3 airborne/ground nodes managing 4 synchronized tuners.
   - Real-time 2D time-frequency waterfall spectrogram, Electronic Order of Battle (EOB) threat library, Pulse Descriptor Word (PDW) stream with CSV/JSON export, and AI vs Legacy Comparison HUD.

## Code Layout
- `src/` (or `hardware/`):
  - `hardware/whittle_engine.hpp` (or `src/rmab_engine.hpp`): C++20 zero-allocation RMAB core.
  - `hardware/dwell_timer.hpp` (or `src/dwell_timer.hpp`): 50µs hardware timing loop harness.
  - `src/bindings.cpp`: pybind11 bindings exposing `rmab_cpp`.
  - `CMakeLists.txt`: CMake build configuration for `rmab_cpp`.
- `schedulers/`:
  - `schedulers/rmab_cpp_wrapper.py`: Drop-in Python wrappers `CppWhittleIndexScheduler` and `CppMultiWhittleIndexScheduler` matching existing interfaces.
  - `schedulers/rmab.py`, `schedulers/multi_schedulers.py`: Existing Python reference implementations.
- `demo/`:
  - `demo/dashboard.py`: Redesigned modern C2-ESM Tactical Operations Center dashboard.
- `tests/`:
  - `tests/test_rmab_cpp_parity.py`: Numerical equivalence tests between C++ and Python schedulers.
  - `tests/test_hardware_timing.py`: Verification of 50µs dwell loop and jitter bounds.
  - `tests/test_cpp_env_integration.py`: Integration of C++ engine inside `DynamicSpectrumEnv` and `MultiReceiverSpectrumEnv`.
  - `tests/test_dashboard_headless.py`: Headless validation of dashboard layout and callbacks.
  - `tests/e2e/`: Requirement-driven E2E test suite (Tiers 1-4).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | C++20 RMAB Closed-Form Index | Closed-form Whittle index computation for K=35 bands without dynamic allocations | M1 | R1, ORIGINAL_REQUEST §13 |
| 2 | C++20 Bayesian Belief Updating | Sensed band likelihood update, unsensed Markov diffusion, AoI increment, online transition learning | M1 | R1, ORIGINAL_REQUEST §14 |
| 3 | Multi-Tuner Top-M Selection | Coordinated collision-free arm selection for M=4 tuners across K=35 bands | M1 | R1, ORIGINAL_REQUEST §13,49 |
| 4 | CMake & pybind11 Build System | CMakeLists.txt compiling clean `rmab_cpp` extension importable in Python | M1 | R1, ORIGINAL_REQUEST §15,41 |
| 5 | High-Res Latency Benchmark | CPU benchmark verifying median decision time <100ns for 35 bands | M1 | R1, ORIGINAL_REQUEST §16,42 |
| 6 | Drop-in Python Wrappers | Drop-in classes matching `WhittleIndexScheduler` and `MultiWhittleIndexScheduler` interfaces | M1 | R1, ORIGINAL_REQUEST §17 |
| 7 | 50µs Dwell Timing Simulation | Hardware dwell loop test harness modeling 50µs slot transitions with monotonic clock | M2 | R2, ORIGINAL_REQUEST §20-22 |
| 8 | Timing Jitter Statistics Export | Measurement and export of median and 99th percentile jitter (<5µs total jitter) to Python | M2 | R2, ORIGINAL_REQUEST §23,45 |
| 9 | Fleet Telemetry Matrix | Live telemetry cards for 3 nodes (UAV-1, UAV-2, Ground Station) with 4-tuner allocations | M3 | R3, ORIGINAL_REQUEST §27,49 |
| 10 | Interactive Multi-Tuner Waterfall | Real-time 2D time-frequency spectrogram showing pulse hits, dwell tracks, and zero-collision indicators | M3 | R3, ORIGINAL_REQUEST §28,51 |
| 11 | EOB Threat Identification Table | Live threat table with emitter ID, type, center freq, estimated PRI, AoI, alert levels | M3 | R3, ORIGINAL_REQUEST §29,50 |
| 12 | PDW Intercept Log & Data Export | Live stream of intercepted PDWs with one-click CSV and JSON downloads | M3 | R3, ORIGINAL_REQUEST §30,50 |
| 13 | AI vs Legacy Comparison HUD | Banner comparing Interception Ratio (IR), Time-to-Intercept (TTI), and throughput | M3 | R3, ORIGINAL_REQUEST §31 |
| 14 | Mathematical Equivalence Tests | Unit tests verifying bit-level/numerical parity between C++ and Python schedulers | M4 | R4, ORIGINAL_REQUEST §34,44 |
| 15 | Environment Integration Tests | Full simulation runs with `rmab_cpp` inside DynamicSpectrumEnv and MultiReceiverSpectrumEnv | M4 | R4, ORIGINAL_REQUEST §35 |
| 16 | Headless Dashboard Verification | Headless WSGI rendering and callback verification without GUI crashes | M4 | R4, ORIGINAL_REQUEST §36,48 |
| 17 | Full Regression Integrity | All existing 87 unit tests continue passing (`uv run pytest`) | M4 | R4, ORIGINAL_REQUEST §54 |
| 18 | Opaque-box E2E Test Suite | Requirement-driven test suite spanning Tiers 1-4 with 100% pass rate | E2E | Project Pattern Dual Track |
| 19 | Adversarial Coverage Hardening | Tier 5 white-box stress testing and forensic audit clean verification | E2E | Project Pattern Final Milestone |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Suite | Opaque-box test suite (Tiers 1-4) derived from user requirements | None | IN_PROGRESS |
| M1 | Zero-Allocation C++20 RMAB Engine & pybind11 | C++20 core, pybind11 module `rmab_cpp`, drop-in Python wrappers, <100ns benchmark | None | PLANNED |
| M2 | 50µs Hardware Dwell Timing Loop Simulation | Real-time 50µs slot simulation, monotonic clock, jitter analytics (<5µs jitter) | M1 | PLANNED |
| M3 | Modern C2-ESM Tactical TOC Dashboard | Redesigned `demo/dashboard.py`, fleet telemetry, waterfall, EOB table, PDW export | None | PLANNED |
| M4 | Automated Verification & Regression Suite | Parity tests, env integration, headless dashboard tests, 87/87 regression | M1, M2, M3 | PLANNED |
| M5 | Final E2E Acceptance & Adversarial Hardening | 100% pass on E2E test suite + Tier 5 adversarial hardening + Forensic Audit | E2E, M4 | PLANNED |

## Interface Contracts
### `rmab_cpp` C++ Core ↔ Python Wrapper
- C++ Class: `WhittleEngine<35>`
  - `void reset()`: Re-initializes belief state to 0.10, AoI to 0, transitions to defaults.
  - `int select_action(float camping_penalty_weight = 0.40f, float aoi_weight = 0.60f)`: Returns best band index $[0, 34]$.
  - `std::array<int, M> select_actions<M>(...)`: Returns top-$M$ unique bands sorted descending by score.
  - `void update_feedback(int band, bool hit)`: Updates belief, AoI, and transition parameters for sensed band; diffuses unsensed bands.
  - `void update_feedback_multi(const std::vector<int>& bands, const std::vector<bool>& hits)`: Batch update for multi-tuner dwells.
  - `std::vector<float> get_beliefs() const`, `std::vector<float> get_aoi() const`, `std::vector<float> get_scores() const`.
- Python Wrapper: `CppWhittleIndexScheduler(BaseScheduler)` and `CppMultiWhittleIndexScheduler(BaseMultiScheduler)`
  - Methods: `reset()`, `select_band() -> int`, `select_bands() -> List[int]`, `update_feedback(band: int, hit: bool)`.

### Hardware Timing Engine ↔ Python Benchmarking
- C++ Class: `DwellTimerSimulation`
  - `void run_dwell_loop(size_t num_slots, double slot_budget_us = 50.0)`
  - `TimingStats get_statistics() const`: Returns struct containing `median_compute_ns`, `p99_compute_ns`, `median_jitter_us`, `p99_jitter_us`, `deadline_misses`.

### C2-ESM Dashboard ↔ Simulation Environments
- Fleet Architecture:
  - Node Alpha (UAV-1): Tuner 0 (Track mode)
  - Node Bravo (UAV-2): Tuners 1 & 2 (FHSS Chase mode)
  - Node Charlie (Ground Station): Tuner 3 (Wideband Sentry mode)
- Data Contracts:
  - Spectrogram Buffer: rolling history of $T=100$ time steps across $K=35$ sub-bands.
  - EOB Table: list of emitter records with keys `emitter_id`, `type`, `freq_ghz`, `pri_us`, `aoi`, `alert_level`.
  - PDW Record: dict/dataclass with `timestamp`, `tuner_id`, `freq_idx`, `freq_ghz`, `rssi_dbm`, `pulse_width_ns`.
