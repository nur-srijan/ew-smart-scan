# Original User Request

## 2026-09-19T07:43:15Z

Deliver a production-ready, zero-allocation C++20 RMAB Whittle Index engine with pybind11 bindings (<100ns latency) and 50µs hardware timing harness, alongside a redesigned modern C2-ESM Tactical Operations Center (TOC) Dashboard with multi-payload fleet telemetry, live RF waterfall, EOB threat library, and PDW export.

Working directory: /Users/nursrijan/dev/sih-project
Integrity mode: development

## Requirements

### R1. Zero-Allocation C++20 RMAB Engine & pybind11 Extension
Implement a high-performance, zero-heap-allocation C++20 engine for the Restless Multi-Armed Bandit (RMAB) Whittle Index scheduler across K=35 sub-bands. The implementation must:
- Update Bayesian belief states and compute closed-form Whittle indices without dynamic memory allocations on the hot path.
- Provide a CMakeLists.txt build system compiling a Python extension module (rmab_cpp) using pybind11.
- Include a high-resolution benchmark harness verifying sub-microsecond latency (<100ns per 35-band ranking decision on CPU).
- Provide a drop-in Python wrapper class matching the existing WhittleIndexScheduler and MultiWhittleScheduler interfaces.

### R2. 50µs Hardware Dwell Timing Loop Simulation
Build a C++ hardware timing test harness that models the strict 50µs dwell timing deadline of radar electronic support systems:
- Simulates real-time receiver slot transitions with monotonic clock timing.
- Logs timing jitter and verifies the scheduler completes well inside the 50µs window.
- Exposes timing statistics (median, 99th percentile jitter) to Python for automated benchmarking.

### R3. C2-ESM Tactical Operations Center Dashboard (demo/dashboard.py)
Redesign demo/dashboard.py into a professional, military-grade C2-ESM (Autonomous Spectrum Surveillance & Telemetry Center) interface featuring:
- Fleet Telemetry Matrix: Live telemetry cards for distributed nodes (Node Alpha UAV-1, Node Bravo UAV-2, Node Charlie Ground Station) displaying operating mode, health status, and 4-tuner frequency allocations.
- Interactive Multi-Tuner Waterfall: Real-time 2D time-frequency spectrogram showing pulse hits, tuner dwell bands, and agile emitter hop tracks with zero tuner collision indicators.
- Electronic Order of Battle (EOB) Threat Table: Live threat identification table displaying emitter ID, type (Fixed, FHSS, Scanning), center frequency, estimated PRI, AoI, and alert level.
- PDW Intercept Log & Data Export: Live stream of intercepted Pulse Descriptor Words with one-click CSV and JSON export for downstream SIGINT analysis.
- AI vs Legacy Comparison HUD: Prominent metrics banner showing real-time gains in Interception Ratio (IR), Time-to-Intercept (TTI), and pulse interception throughput.

### R4. Automated Verification & Regression Suite
- Unit tests verifying numerical equivalence between C++ rmab_cpp and Python ew_sim schedulers.
- End-to-end integration test running rmab_cpp inside DynamicSpectrumEnv and MultiReceiverSpectrumEnv.
- Headless verification ensuring demo/dashboard.py loads and renders components without exceptions.

## Acceptance Criteria

### Edge C++ Performance & Correctness
- [ ] rmab_cpp builds cleanly via CMake/pybind11 and is importable in the virtual environment (uv run python -c "import rmab_cpp").
- [ ] Median C++ decision time per 35-band schedule is verified < 100ns in the benchmark suite.
- [ ] Zero memory allocations occur during the step() / select_action() cycle on the hot path.
- [ ] 100% mathematical parity with Python WhittleIndexScheduler across identical observation vectors.
- [ ] 50µs dwell timing harness reports timing margins with deterministic completion (<5µs total jitter).

### Tactical TOC Dashboard Usability
- [ ] demo/dashboard.py launches cleanly using uv run streamlit run demo/dashboard.py (or uv run demo/dashboard.py).
- [ ] Multi-payload fleet view displays 3 distinct nodes with synchronized 4-tuner allocation.
- [ ] EOB Threat Library table updates in real time and PDW CSV export button produces valid downloaded files.
- [ ] 0.0% tuner collisions guaranteed and indicated on the UI telemetry HUD.

### Project & Test Integrity
- [ ] All existing 87 unit tests continue to pass (uv run pytest).
- [ ] New unit tests covering the C++ extension and dashboard engine pass without warnings or errors.
