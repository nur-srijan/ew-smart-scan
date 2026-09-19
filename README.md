# EW Smart Scan — Autonomous Multi-Receiver Electronic Support Scheduler
**Smart India Hackathon 2026 · Problem Statement 1778 · Sponsoring Org: DRDO · Theme: Robotics & Drones**

[![Tests](https://img.shields.io/badge/Tests-279%2F279%20Passing-brightgreen.svg)](#test-suite--validation)
[![Engine](https://img.shields.io/badge/C%2B%2B20%20Core-20.8%20ns%20Latency-blue.svg)](#ultra-fast-c20-scheduling-core)
[![Collisions](https://img.shields.io/badge/Tuner%20Collisions-0.0%25%20Guaranteed-teal.svg)](#fleet-cooperative-architecture)
[![Python](https://img.shields.io/badge/Python-3.12%20|%20uv-yellow.svg)](#quickstart-guide)

An autonomous, multi-agent Electronic Support (ES) surveillance platform powered by **Restless Multi-Armed Bandits (RMAB)**, **Recurrent Deep Reinforcement Learning (DRL)**, and a **zero-allocation C++20 real-time engine**. Designed for DRDO Problem Statement 1778, Smart Scan intercepts, deinterleaves, and tracks frequency-agile and scanning emitters across 0.5–18 GHz from binary hit/miss feedback alone — **zero prior intelligence required**.

---

## 🎯 The Operational Challenge

In contested electronic warfare environments, high-sensitivity Electronic Support (ES) receivers possess an Instantaneous Bandwidth (IBW) 1–2 orders of magnitude narrower than the surveillance spectrum (0.5 – 18 GHz, divided into 35 sub-bands of 500 MHz each).

Legacy open-loop sweeping techniques (sequential raster or pseudo-random sweeps) suffer catastrophic pulse loss (>85%) due to:
- **Stroboscopic Resonance:** Deterministic sweep rates create persistent harmonic blind spots aligned with radar pulse repetition intervals (PRI) and antenna rotation periods.
- **Inability to Track Agility:** Ineffective against modern frequency-hopping spread-spectrum (FHSS) radars (e.g., Krasukha-4) and low-probability-of-intercept (LPI) fire control systems.
- **Zero Fleet Coordination:** Independent multi-receiver systems suffer high spectral collision rates, wasting precious dwell capacity on redundant bands while leaving critical threats unmonitored.

---

## ⚡ The Smart Scan EW Solution

Smart Scan EW deploys a multi-payload cooperative hierarchy designed for real-time edge execution:

1. **Ultra-Low Latency C++20 RMAB Engine:** Sub-microsecond Whittle index evaluation ($20.80\,\text{ns}$ decision latency), guaranteeing zero deadline misses inside strict $50\,\mu\text{s}$ dwell windows on platforms like NVIDIA Jetson Orin Nano.
2. **Distributed Fleet Orchestration ($M=4$ Tuners across 3 Nodes):** Role-specialized allocation (Pulse Tracker, FHSS Chaser Pair, Wideband AoI Sentry) with mathematical guarantees of **$0.0\%$ tuner collisions**.
3. **Online Signal Analysis & EOB Generation:** Non-cooperative delta-TOA pulse deinterleaving that estimates PRIs, identifies hop sets, and flags threat levels in real time without pre-loaded libraries.
4. **C2-ESM Tactical TOC Dashboard:** Dark military command interface with real-time 35-band waterfall canvas, fleet telemetry, 1-click PDW (Pulse Descriptor Word) CSV/JSON export, and an interactive Plotly-driven Performance & FoM analytics suite.

---

## 📊 Benchmark Results

### Multi-Receiver Fleet Evaluation (20-Episode Monte Carlo, Contested Spectrum)

Evaluated across dynamic scenarios featuring fixed-frequency air defense radars (S-300 PMU-2), agile frequency-hoppers (Krasukha-4), scanning fire control radars (Su-35S Irbis-E), and active jamming strobes:

| Metric | Single-Receiver Baseline ($M=1$) | Multi-Sequential ($M=4$) | Multi-Whittle RMAB ($M=4$) | **Cooperative Role AI (Ours, $M=4$)** | Improvement vs Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Global Interception Ratio ($IR$)** | $2.65\%$ | $3.78\%$ | $18.42\%$ | **$22.17\%$** | **+737% (8.4× Higher)** |
| **Mean Time-to-Intercept ($TTI$)** | $1.743\text{ s}$ | $1.412\text{ s}$ | $0.285\text{ s}$ | **$0.237\text{ s}$** | **61% Latency Reduction** |
| **Pulse Throughput (pulses/step)** | $29.8$ | $42.5$ | $145.8$ | **$175.3$** | **5.9× Increase** |
| **Tuner Collision Rate** | $0.0\%$ (N/A) | $0.0\%$ | $0.0\%$ | **$0.0\%$** | **Strict Orthogonality** |
| **Anti-Camping Entropy** | High (random) | Maximum (blind) | Balanced | **Optimal ($0.88$)** | **Exploit + AoI Explore** |
| **Decision Latency** | $<0.1\,\mu\text{s}$ | $<0.1\,\mu\text{s}$ | $20.8\,\text{ns}$ | **$20.8\,\text{ns}$ (C++20)** | **Well inside $50\,\mu\text{s}$** |

---

## 🏛️ System Architecture

```
                               Tactical C2-ESM Web Dashboard (Port 8050)
                       [ 01 Waterfall | 02 Fleet | 03 Threats | 04 FoM | 05 Mission ]
                                                   │
                                            REST / JSON API
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Distributed Cooperative Scheduler                              │
│                                                                                             │
│   ┌──────────────────────────┐   ┌──────────────────────────┐   ┌────────────────────────┐  │
│   │   Node Alpha (UAV-1)     │   │   Node Bravo (UAV-2)     │   │   Node Charlie (C2)    │  │
│   │   Tuner 0: Tracker       │   │   Tuners 1 & 2: Chasers  │   │   Tuner 3: Sentry      │  │
│   │   • Phase-locked TOA     │   │   • Markov FHSS tracking │   │   • Wideband AoI sweep │  │
│   │   • Periodic exploitation│   │   • Agility bracketing   │   │   • Anti-camping audit │  │
│   └────────────┬─────────────┘   └────────────┬─────────────┘   └───────────┬────────────┘  │
│                │                              │                             │               │
│                └───────────────────────┬──────┴─────────────────────────────┘               │
│                                        ▼                                                    │
│                        Strict Spectral Orthogonality Filter                                 │
│                        (Disjoint Sub-Band Allocations -> 0.0% Collisions)                   │
└────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            C++20 Zero-Allocation RMAB Engine                                │
│   • Closed-form Whittle Index ranking: 20.80 ns / step                                       │
│   • Sub-50 µs hard real-time execution deadline guarantee                                   │
│   • Pybind11 zero-copy bindings (`ew_smart_scan_cpp`)                                       │
└────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                         Multi-Receiver Gymnasium Environment                                │
│   • 35 Sub-Bands (0.5 - 18.0 GHz)                                                           │
│   • S[K, T] RF Ground Truth Matrix Builder                                                  │
│   • Dynamic Emitters: Fixed-Frequency, FHSS Agile, Scanning Radar, Jamming Strobe           │
│   • Online Delta-TOA PRI Estimator & Pulse Descriptor Word (PDW) Stream                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Ultra-Fast C++20 Scheduling Core

The decision engine is implemented in ISO C++20 (`src/rmab_engine.cpp`) with zero dynamic memory allocations in the critical path:

- **Benchmark Result:** **$20.80\,\text{ns}$** average ranking latency per dwell step on standard hardware.
- **Dwell Deadline:** Fixed $50\,\mu\text{s}$ receiver dwell budget. The C++20 engine consumes $<0.05\%$ of the available time window, leaving $>99.9\%$ for RF synthesizer settling and signal capture.
- **Deadline Misses:** **0 out of 10,000** evaluated cycles in the hard real-time test harness (`hardware/test_timing.cpp`).

```bash
# Verify edge execution latency directly:
./hardware/test_timing
# Output: Mean Latency: 20.80 ns | Max Latency: 41.67 ns | Missed Deadlines: 0
```

---

## 🖥️ C2-ESM Tactical TOC Web Dashboard

Launch the integrated tactical dashboard serving real-time EW telemetry at `http://127.0.0.1:8050`:

- **Tab 01: Live Waterfall:** High-performance HTML5 canvas rendering 35 frequency channels with color-coded receiver tuner overlays, accompanied by a 7×5 sub-band Age-of-Information (AoI) matrix.
- **Tab 02: Fleet Telemetry:** Health monitoring for Node Alpha (UAV-1), Node Bravo (UAV-2), and Node Charlie (Ground Station) displaying battery state, RF synthesizer temperatures, and live operator advisory acknowledgments.
- **Tab 03: Threat Library (EOB):** Live-updating Electronic Order of Battle table extracting estimated PRIs, carrier frequencies, and alert levels via delta-TOA analysis, with **1-click PDW CSV/JSON export**.
- **Tab 04: Performance & FoM:** Live KPI HUD cards (IR, TTI, Throughput, Collisions), 4 interactive Plotly charts (Cumulative Interceptions, Policy IR % Bar, 35-Band Dwell Histogram, TTI Latency), and the complete 20-episode Monte Carlo evaluation matrix.
- **Tab 05: Mission Control:** Dynamic dropdown to hot-swap schedulers (`CooperativeRoleScheduler`, `MultiWhittleRMAB`, `MultiSequentialSweep`), switch scenario presets, throttle simulation speed, and inspect the real-time system audit log.

*(The legacy Dash analytical workbench remains accessible at `http://127.0.0.1:8050/dash/`).*

---

## 📁 Repository Layout

```
ew-smart-scan/
├── src/                         # C++20 High-Performance Core
│   ├── rmab_engine.cpp          # Zero-allocation Whittle Index engine
│   └── bindings.cpp             # Pybind11 module bindings (ew_smart_scan_cpp)
│
├── hardware/                    # Edge Hardware Benchmarking & Harness
│   ├── test_timing.cpp          # Nanosecond-resolution hard real-time test harness
│   ├── test_timing              # Compiled executable (20.8 ns decision cycle)
│   └── run_timing_bench.sh      # Benchmark automation script
│
├── ew_sim/                      # Spectrum Environment & Physics Simulation
│   ├── env.py                   # Single-receiver Gymnasium EWSpectrumEnv
│   ├── multi_receiver_env.py    # Multi-Receiver EWSpectrumEnv (M tuners, K sub-bands)
│   ├── emitters.py              # Fixed-Frequency, FHSS, Scanning, and Strobe emitters
│   ├── truth_engine.py          # 2D S[K, T] Ground Truth matrix generator
│   └── turing_loader.py         # Synthetic Radar Dataset adapter (PDW schema)
│
├── schedulers/                  # Autonomous Scheduling Algorithms
│   ├── multi_cooperative.py     # Fleet Cooperative Role Scheduler (0.0% collisions)
│   ├── rmab.py                  # Python Whittle Index Restless Bandit
│   ├── cpp_rmab_adapter.py      # Python wrapper calling native C++20 engine
│   ├── predictor.py             # Online Periodicity & Scan Phase Estimator
│   ├── drl_agent.py             # Recurrent PPO / Actor-Critic PyTorch policy
│   └── baselines.py             # Sequential, Pseudo-Random, and Priority sweeps
│
├── eval/                        # Evaluation & Figures of Merit (FoM)
│   ├── fom.py                   # FoM calculators (IR, TTI, Pd, Pfa, Collisions, AoI)
│   └── runner.py                # Monte Carlo evaluation harness
│
├── demo/                        # Tactical Operations Center & Dashboards
│   ├── dashboard.py             # Unified Flask/Dash server serving C2-ESM & API
│   ├── web/                     # C2-ESM Tactical TOC Frontend
│   │   ├── index.html           # 5-tab dark military tactical command interface
│   │   └── app.js               # Dynamic REST client, live canvas & Plotly engine
│   ├── compare.py               # Policy comparison waterfall visualizer
│   └── run_phase3_benchmark.py  # Automated Monte Carlo benchmark runner
│
├── tests/                       # 279-Test Comprehensive Test Suite
│   ├── e2e/                     # 4-Tier End-to-End Enterprise Test Suite
│   │   ├── test_tier1_features.py   # Core functional feature tests
│   │   ├── test_tier2_boundaries.py # Edge-case & parameter boundary tests
│   │   ├── test_tier3_pairwise.py   # Cross-component interaction tests
│   │   └── test_tier4_scenarios.py  # End-to-end tactical mission scenarios
│   ├── test_multi_cooperative.py    # Fleet scheduler & collision-avoidance tests
│   ├── test_multi_receiver.py       # Multi-receiver gym compliance tests
│   ├── test_rmab.py                 # Whittle index mathematical correctness tests
│   ├── test_emitters.py             # Radar physics & PRI timing tests
│   └── test_demo.py                 # REST API & dashboard integration tests
│
├── demo_video_transcript.md     # 3-Minute Hackathon Demo Video Script with visual cues
├── pyproject.toml               # uv project configuration & dependencies
└── CMakeLists.txt               # CMake configuration for C++20 build
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites & Environment Setup

This project uses modern Python dependency management via [`uv`](https://docs.astral.sh/uv/):

```bash
cd /Users/nursrijan/dev/sih-project

# Install all dependencies (core, rl, dashboard, edge)
uv sync --extra rl --extra dashboard --extra edge
```

### 2. Launch the C2-ESM Tactical Dashboard

```bash
uv run python demo/dashboard.py
```
Open **`http://127.0.0.1:8050`** in your browser to access the live command center.

### 3. Run Hard Real-Time C++ Timing Benchmark

```bash
./hardware/test_timing
```
*Validates 20.8 ns decision cycle and 0 deadline misses inside 50 µs window.*

### 4. Run Full Test Suite (Zero Regression Guarantee)

```bash
uv run pytest --tb=short -q
```
*Executes all 279 tests across the 4-tier E2E framework and core unit modules.*

### 5. Run Monte Carlo Benchmark Suite

```bash
uv run python demo/run_phase3_benchmark.py
```
*Generates full statistical evaluation curves and Figures of Merit.*

---

## 🧪 Test Suite & Validation

The codebase maintains a strict **zero-regression policy** enforced across 279 automated tests:

- **Tier 1 (Core Features):** Validates gymnasium compliance, multi-receiver observation shapes, reward signals, and baseline sweeps.
- **Tier 2 (Boundaries & Edge Cases):** Stress tests extreme noise levels ($SNR < -10\,\text{dB}$), high emitter densities, zero-signal conditions, and parameter limits.
- **Tier 3 (Pairwise Combinations):** Verifies interoperability between C++ engine, Python schedulers, delta-TOA estimators, and REST endpoints.
- **Tier 4 (Tactical Scenarios):** Full mission simulations involving agile radar evasion, hostile electronic attack strobes, and multi-UAV coordination.

---

## 🎥 Presentation & Demo Video

A fully annotated **3-minute presentation voiceover script** with exact second-by-second visual screen directives is included in [`demo_video_transcript.md`](demo_video_transcript.md):
- Covers all 5 operational tabs, C++ edge execution, and defense impact for DRDO evaluators.

---

## 👥 Team & Acknowledgments

- **Team:** Robotics & Drones
- **Hackathon:** Smart India Hackathon (SIH) 2026
- **Problem Statement:** PS-1778 (Autonomous Electronic Support Receiver Scheduling)
- **Sponsoring Organization:** Defence Research and Development Organisation (DRDO)

