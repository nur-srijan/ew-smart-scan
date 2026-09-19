# Dispatch Record

## 2026-09-19T07:44:09Z

You are the Project Orchestrator for the EW SmartScan project.

Repository Root: /Users/nursrijan/dev/sih-project
Authoritative Request: /Users/nursrijan/dev/sih-project/.agents/ORIGINAL_REQUEST.md
Your Working Directory: /Users/nursrijan/dev/sih-project/.agents/teamwork_preview_orchestrator_1 (also mirrored at /Users/nursrijan/dev/siiiiih/.agents/teamwork_preview_orchestrator_1)

Mission:
Deliver a production-ready, zero-allocation C++20 RMAB Whittle Index engine with pybind11 bindings (<100ns latency) and 50µs hardware timing harness, alongside a redesigned modern C2-ESM Tactical Operations Center (TOC) Dashboard with multi-payload fleet telemetry, live RF waterfall, EOB threat library, and PDW export.

Detailed Requirements:
R1. Zero-Allocation C++20 RMAB Engine & pybind11 Extension
- Implement high-performance, zero-heap-allocation C++20 engine for Restless Multi-Armed Bandit (RMAB) Whittle Index scheduler across K=35 sub-bands.
- Closed-form Whittle indices & Bayesian belief updates with ZERO dynamic memory allocations on the hot path (step / select_action).
- CMakeLists.txt compiling Python extension module (rmab_cpp) using pybind11.
- High-resolution benchmark harness verifying sub-microsecond latency (<100ns per 35-band ranking decision on CPU).
- Drop-in Python wrapper class matching existing WhittleIndexScheduler and MultiWhittleScheduler interfaces.

R2. 50µs Hardware Dwell Timing Loop Simulation
- Build C++ hardware timing test harness modeling strict 50µs dwell timing deadline of radar electronic support systems.
- Simulates real-time receiver slot transitions with monotonic clock timing.
- Logs timing jitter and verifies scheduler completes well inside the 50µs window (<5µs total jitter).
- Exposes timing statistics (median, 99th percentile jitter) to Python for automated benchmarking.

R3. C2-ESM Tactical Operations Center Dashboard (demo/dashboard.py)
- Redesign demo/dashboard.py into professional military-grade C2-ESM interface.
- Fleet Telemetry Matrix: live telemetry cards for distributed nodes (Node Alpha UAV-1, Node Bravo UAV-2, Node Charlie Ground Station) displaying operating mode, health status, and 4-tuner frequency allocations.
- Interactive Multi-Tuner Waterfall: real-time 2D time-frequency spectrogram showing pulse hits, tuner dwell bands, agile emitter hop tracks with zero tuner collision indicators.
- EOB Threat Table: live threat identification table displaying emitter ID, type (Fixed, FHSS, Scanning), center frequency, estimated PRI, AoI, alert level.
- PDW Intercept Log & Data Export: live stream of intercepted Pulse Descriptor Words with one-click CSV and JSON export.
- AI vs Legacy Comparison HUD: prominent metrics banner showing real-time gains in Interception Ratio (IR), Time-to-Intercept (TTI), pulse interception throughput.

R4. Automated Verification & Regression Suite
- Unit tests verifying numerical equivalence between C++ rmab_cpp and Python ew_sim schedulers.
- End-to-end integration test running rmab_cpp inside DynamicSpectrumEnv and MultiReceiverSpectrumEnv.
- Headless verification ensuring demo/dashboard.py loads and renders components without exceptions.
- All existing 87 unit tests must continue to pass (uv run pytest).

Protocol:
- Initialize your BRIEFING.md, plan.md, and progress.md in your working directory.
- Keep progress.md regularly updated as milestones complete.
- When all requirements and acceptance criteria are completed and tested, report completion to the Sentinel.
