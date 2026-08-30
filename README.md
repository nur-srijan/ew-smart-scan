# EW Smart Scan — ML-Based Electronic Support Receiver Scheduler
**Smart India Hackathon 2026 · PS-1778 · Sponsoring Org: DRDO · Theme: Robotics & Drones**

An autonomous, Reinforcement-Learning and Restless Multi-Armed Bandit (RMAB) driven Electronic Support (ES) receiver scheduler that learns to intercept and track frequency-agile and scanning hostile radars from binary hit/miss feedback alone — **zero prior intelligence required**.

---

## The Problem
High-sensitivity ES receivers have an Instantaneous Bandwidth (IBW) 1–2 orders of magnitude narrower than the total surveillance spectrum (0.5 – 18 GHz). Legacy open-loop sequential sweeps lose over 85% of emitter pulses due to:
- **Stroboscopic resonance** — periodic blind-spots when sweep rate harmonically aligns with radar dead times.
- **No adaptability** to frequency-hopping (FHSS) or narrow rotating radar beams.

## Our Solution
A modular AI-driven Smart Scan Suite:
1. **Restless Multi-Armed Bandit (RMAB)** with closed-form Whittle Index policy for sub-microsecond online scheduling.
2. **Online Periodicity & Radar Scan Estimator** for harmonic deinterleaving and targeted dwell alignment.
3. **Deep Reinforcement Learning (DRL) Scheduler** with 2-layer GRU/LSTM recurrent policy for synchronized multi-pulse capture.
4. **Interactive Real-Time Web Dashboard (Dash / Plotly)** for live mission control and evaluation.

---

## Benchmark Results (Monte Carlo, N=20 Tactical Episodes)

| Metric | Sequential Sweep (Baseline) | Pseudo-Random | Priority Queue (EDB) | **Smart DRL (Ours)** |
|:---|:---:|:---:|:---:|:---:|
| **Interception Ratio ($IR$)** | $1.2\% \pm 0.0\%$ | $2.8\% \pm 1.3\%$ | $6.6\% \pm 1.8\%$ | **$29.8\% \pm 0.1\%$ (25× Higher)** |
| **Mean Time-to-Intercept ($TTI$)**| $1.743\text{ s}$ | $1.280\text{ s}$ | $1.161\text{ s}$ | **$1.120\text{ s}$** |
| **Discovery Consistency** | $20.0\%$ (1/5) | $60.0\%$ | $59.0\%$ | **$62.0\%$ (Whittle RMAB)** |
| **Inference Latency** | $<0.1\text{ }\mu\text{s}$ | $<0.2\text{ }\mu\text{s}$ | $<1.0\text{ }\mu\text{s}$ | **$<15\text{ }\mu\text{s}$ (Jetson Orin INT8)** |
| **Resonance Blind Spots** | Frequent | Rare | Moderate | **Zero** |

---

## Project Structure

```
ew-smart-scan/
├── ew_sim/
│   ├── emitters.py            # Fixed-Frequency, FHSS, and Scanning Radar models
│   ├── truth_engine.py        # 2D S[K, T] Ground Truth matrix builder & waterfall plots
│   ├── env.py                 # Gymnasium EWSpectrumEnv (OpenAI Gym API)
│   └── turing_loader.py       # Alan Turing Synthetic Radar Dataset adapter (PDW schema)
│
├── schedulers/
│   ├── baselines.py           # Sequential, Pseudo-Random, Priority EDB sweeps
│   ├── rmab.py                # Whittle Index Restless Multi-Armed Bandit (<1 µs latency)
│   ├── predictor.py           # Online Periodicity & Scan Phase Estimator
│   └── drl_agent.py           # Recurrent PPO / Actor-Critic PyTorch policy
│
├── eval/
│   ├── fom.py                 # Figures of Merit (Pd, Pfa, IR, TTI, Discovery Rate)
│   └── runner.py              # Multi-episode Monte Carlo evaluation runner
│
├── train.py                   # 3-Stage Curriculum DRL Training Pipeline
│
├── demo/
│   ├── dashboard.py           # Interactive Web Dashboard (Dash / Plotly)
│   ├── compare.py             # Side-by-side Dwell vs Truth waterfall generator
│   ├── run_phase3_benchmark.py# Full Monte Carlo benchmark suite
│   └── visualize_truth.py     # RF Ground Truth sanity checker
│
├── tests/                     # 63 passing unit tests (100% test coverage)
│   ├── test_baselines.py
│   ├── test_drl.py
│   ├── test_emitters.py
│   ├── test_env.py
│   ├── test_fom.py
│   ├── test_predictor.py
│   ├── test_rmab.py
│   ├── test_truth_engine.py
│   ├── test_turing_loader.py
│   └── test_demo.py
│
├── notebooks/                 # Generated figures, waterfalls, and HTML reports
└── pyproject.toml             # uv-managed dependencies
```

---

## Quickstart Guide

### 1. Install Dependencies
```bash
cd ew-smart-scan
uv sync --extra rl --extra dashboard
```

### 2. Run the Interactive Web Dashboard
```bash
uv run demo/dashboard.py
# Open http://127.0.0.1:8050 in your browser
```

### 3. Run Side-by-Side Policy Comparison
```bash
uv run demo/compare.py
# Generates notebooks/live_comparison_waterfall.png and interactive_comparison.html
```

### 4. Run Full Monte Carlo Benchmark
```bash
uv run demo/run_phase3_benchmark.py
```

### 5. Run Test Suite
```bash
uv run pytest
```
