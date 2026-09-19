## 2026-09-19T07:52:24Z

You are Worker M1 (C++20 RMAB Engine & pybind11 Specialist).
Your working directory is: /Users/nursrijan/dev/sih-project/.agents/teamwork_preview_worker_m1
Authoritative Request: /Users/nursrijan/dev/sih-project/.agents/ORIGINAL_REQUEST.md

You MUST read /Users/nursrijan/dev/sih-project/.agents/ORIGINAL_REQUEST.md before starting work.
Also read:
- /Users/nursrijan/dev/sih-project/PROJECT.md
- /Users/nursrijan/dev/sih-project/.agents/teamwork_preview_explorer_survey_1/report.md
- /Users/nursrijan/dev/sih-project/.agents/teamwork_preview_explorer_survey_2/report.md

Exclusive Write Ownership:
- hardware/whittle_engine.hpp
- hardware/dwell_timer.hpp
- src/bindings.cpp
- CMakeLists.txt
- schedulers/rmab_cpp_wrapper.py
- benchmarks/benchmark_rmab.py
- hardware/test_timing.cpp

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Mission:
Deliver Requirement R1 (Zero-Allocation C++20 RMAB Engine & pybind11 Extension) and Requirement R2 (50µs Hardware Dwell Timing Loop Simulation):
1. Zero-Allocation C++20 Engine (hardware/whittle_engine.hpp):
   - K=35 sub-bands fixed size.
   - Closed-form Whittle index computation: W(p) = (p * Delta + P01) / (1 - Delta + p * Delta) where Delta = P11 - P01.
   - Composite score with AoI bonus (0.60 * min(1.0, AoI / 50.0)), anti-camping penalty (0.40 * consecutive_dwells), tie-breaker.
   - Bayesian belief update: sensed band update with Pd=0.95, Pfa=1e-4, duty=0.20, clamped to [0.001, 0.999]; unsensed bands diffusion with alpha=0.02, prior=0.15.
   - Online learning of transition probabilities P01, P11 with learning rate eta=0.05 when dwell interval <= 10.
   - Multi-tuner Top-M arm selection (M=4) selecting top distinct arms descending by score, guaranteeing 0.0% collisions.
   - Absolutely ZERO dynamic heap allocation (no malloc/new) on the hot path (step, select_action, update_feedback). Use std::array, fixed stack buffers, or static storage.
2. CMakeLists.txt & pybind11 bindings (src/bindings.cpp):
   - Build system using pybind11 compiling `rmab_cpp` extension module for Python 3.12.
   - Build extension in-place or into site-packages so `uv run python -c "import rmab_cpp"` succeeds cleanly.
   - Expose both single-tuner and multi-tuner C++ engine interfaces and dwell timing harness to Python.
3. Drop-in Python Wrappers (schedulers/rmab_cpp_wrapper.py):
   - Provide `CppWhittleIndexScheduler` and `CppMultiWhittleIndexScheduler` inheriting from `BaseScheduler` and `BaseMultiScheduler` respectively.
   - Drop-in compatibility with existing `WhittleIndexScheduler` and `MultiWhittleIndexScheduler`.
4. High-Resolution Latency Benchmark (benchmarks/benchmark_rmab.py):
   - Benchmark decision latency on CPU over 100,000 cycles.
   - Verify median decision time is strictly < 100ns per 35-band schedule.
5. 50µs Hardware Dwell Timing Loop Simulation (hardware/dwell_timer.hpp):
   - Monotonic clock timing (`std::chrono::steady_clock`) simulating 50µs dwell timing window.
   - Measures compute time and logs timing jitter across >=2000 slots.
   - Verifies zero deadline misses and total jitter < 5µs.
   - Exposes timing statistics (median jitter, p99 jitter, deadline violations) to Python via `rmab_cpp`.

Verification:
- Compile `rmab_cpp` via cmake/compiler command.
- Test import: `uv run python -c "import rmab_cpp; print(rmab_cpp.__doc__)"`.
- Run benchmark script and record latency metrics.
- Ensure all 87 existing tests still pass: `uv run pytest`.

Deliverables:
Write detailed implementation report to /Users/nursrijan/dev/sih-project/.agents/teamwork_preview_worker_m1/report.md and handoff to /Users/nursrijan/dev/sih-project/.agents/teamwork_preview_worker_m1/handoff.md.
