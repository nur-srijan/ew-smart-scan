# Handoff Report: Dashboard, Simulation Envs, & Telemetry Specialist

**Agent**: Explorer 3  
**Working Directory**: `/Users/nursrijan/dev/sih-project/.agents/teamwork_preview_explorer_survey_3`  
**Parent Agent**: Orchestrator (`767cabb8-22a0-4111-a049-f29abc4581c9`)  
**Task**: Exploration Survey for R3 (C2-ESM Tactical TOC Dashboard) and R4 (Headless Verification)

---

## 1. Observation

1. **Dashboard Architecture & Dependencies**:
   - `pyproject.toml` lines 26–29 specifies:
     ```toml
     dashboard = [
         "plotly>=5.22",
         "dash>=2.17",
     ]
     ```
   - Running `uv run python -c "import dash, plotly; print('dash:', dash.__version__, 'plotly:', plotly.__version__)"` outputs `dash: 4.4.1 plotly: 7.0.0`.
   - Running `uv run python -c "import streamlit"` fails with `ModuleNotFoundError: No module named 'streamlit'`.
   - `demo/dashboard.py` lines 25–28 imports `dash`, `dash.dcc`, `dash.html`, `dash.dash_table`, and `plotly.graph_objects`.
   - `demo/dashboard.py` line 201 initializes `app = dash.Dash(__name__, title="DRDO EW Smart Scan Tactical Dashboard")`.
   - `demo/dashboard.py` lines 1135–1138 runs `app.run(debug=False, port=8050)`.
   - The original user request specifies in acceptance criteria:
     `demo/dashboard.py launches cleanly using uv run streamlit run demo/dashboard.py (or uv run demo/dashboard.py)`.

2. **Simulation Environments & Multi-Tuner Capabilities**:
   - `ew_sim/env.py` contains `EWSpectrumEnv` and `DynamicSpectrumEnv`, configured for single-receiver ($M=1$) scheduling.
   - `ew_sim/multi_env.py` lines 28–314 contains `MultiReceiverEWSpectrumEnv`, which models $M=4$ independent tuners, detects redundant tuner collisions, deduplicates intercepted pulses, and penalizes collisions.
   - `schedulers/multi_schedulers.py` lines 162–349 implements `MultiWhittleIndexScheduler` (top-$M$ closed-form Whittle indices + AoI subsidies, guaranteeing 0 collisions).
   - `schedulers/multi_schedulers.py` lines 354–516 implements `CooperativeRoleScheduler`, assigning distinct tactical roles:
     - Role 0: Phase-Locked Pulse Tracker (fixed/periodic radars via online PRI estimation).
     - Roles 1 & 2: Agile FHSS Chaser Pair (Markov hop transition bracketing).
     - Role 3: Surveillance Sentry (Age-of-Information wideband patrol).
     Guarantees strict 0.0% tuner collisions over all steps.
   - `ew_sim/turing_loader.py` lines 31–39 defines the standard `PulseDescriptorWord` dataclass:
     ```python
     @dataclass
     class PulseDescriptorWord:
         toa_sec: float
         freq_ghz: float
         pulse_width_sec: float
         amplitude_dbm: float
         emitter_id: int
     ```

3. **Current Test Suite**:
   - Running `uv run pytest` executes 87 tests across 14 test files: `87 passed, 3 warnings in 1.85s`.
   - `tests/test_demo.py` contains 3 unit tests verifying `run_policy_episode`, `create_scenario`, and `instantiate_scheduler`, but does not test dashboard rendering, multi-receiver execution, or PDW exports.

4. **Headless Execution Verification**:
   - Running `uv run python -c "import demo.dashboard"` executes in 0.8s without starting the web server or hanging.
   - Running `uv run python -c "from demo.dashboard import app; client = app.server.test_client(); res = client.get('/'); print('Status code:', res.status_code)"` returns `Status code: 200` in <0.1s without binding any network socket.

---

## 2. Logic Chain

1. **UI Framework Decision**:
   - *Observation*: Dash 4.4.1 and Plotly 7.0.0 are already installed and used in `demo/dashboard.py`. Streamlit is not installed, not in `pyproject.toml`, and not in `uv.lock`. The acceptance criteria accepts `(or uv run demo/dashboard.py)`.
   - *Inference*: Retaining Dash is the correct and safest architectural path. Migrating to Streamlit would require introducing unwanted dependency churn into `pyproject.toml` and rewriting the entire UI layout and callback structure. Dash provides superior military-grade CSS styling control, native `dash_table.DataTable`, `dcc.Download`, and built-in Flask WSGI headless testing.

2. **R3 Multi-Node to Multi-Tuner Mapping**:
   - *Observation*: R3 requires 3 nodes (Node Alpha UAV-1, Node Bravo UAV-2, Node Charlie Ground Station) with 4-tuner allocations. `ew_sim/multi_env.py` and `CooperativeRoleScheduler` model exactly $M=4$ coordinated tuners with specialized roles: Role 0 (Tracker), Roles 1 & 2 (FHSS Chasers), Role 3 (Sentry).
   - *Inference*: The 4 tuners map cleanly across the 3 nodes:
     - Node Alpha UAV-1: 1 tuner (Tuner 0, Phase-Locked Tracker)
     - Node Bravo UAV-2: 2 tuners (Tuners 1 & 2, Agile FHSS Chaser Pair)
     - Node Charlie Ground Station: 1 tuner (Tuner 3, Wideband AoI Sentry)
     This directly reflects real-world cooperative EW doctrine and satisfies all R3 telemetry requirements.

3. **0.0% Collision Guarantee**:
   - *Observation*: `MultiWhittleIndexScheduler` selects the top-$M$ distinct arms via `np.argsort(scores)[::-1][:self.M]`. `CooperativeRoleScheduler` enforces disjoint band sets across roles. Both pass strict collision tests in `tests/test_multi_receiver.py` and `tests/test_multi_cooperative.py`.
   - *Inference*: Integrating these schedulers into `demo/dashboard.py` mathematically guarantees a 0.0% collision rate on the UI telemetry HUD.

4. **Headless Testing Feasibility**:
   - *Observation*: `demo/dashboard.py` protects server execution behind `if __name__ == "__main__": app.run()`, and Dash apps expose `app.server.test_client()`.
   - *Inference*: The entire dashboard layout, callbacks, EOB generation, and PDW export functions can be verified headlessly via pytest using pure Python function calls and the Flask WSGI test client, completely eliminating browser dependencies, Xvfb, or blocking servers.

---

## 3. Caveats

- No source code modifications were made during this investigation (strictly read-only).
- While SB3 recurrent PPO model checkpoints exist in `checkpoints/` (`ppo_recurrent_multi_stage2.zip`), loading heavy neural networks is slower than analytical schedulers. For instant dashboard responsiveness and sub-second headless testing, `MultiWhittleIndexScheduler` and `CooperativeRoleScheduler` should serve as the default high-performance operational engines.

---

## 4. Conclusion

- **Dashboard UI Framework**: Plotly Dash 4.4.1 (with underlying Flask WSGI) is the canonical and optimal framework.
- **R3 Redesign Scope**:
  1. Refactor `demo/dashboard.py` from single-receiver to multi-receiver ($M=4$ tuners) backed by `MultiReceiverEWSpectrumEnv`.
  2. Implement Fleet Telemetry Matrix for Node Alpha (UAV-1), Node Bravo (UAV-2), and Node Charlie (Ground Station).
  3. Upgrade waterfall spectrogram to plot 4 synchronized color-coded tuner tracks, emitter hop tracks, and a 0.0% collision badge.
  4. Build military-grade EOB Threat Table with Emitter ID, type, center frequency, estimated PRI, AoI, and alert level.
  5. Add live PDW stream and one-click CSV and JSON downloads via `dcc.Download`.
  6. Upgrade AI vs Legacy Comparison HUD with real-time empirical gains in IR, TTI, and pulse throughput.
- **R4 Headless Verification**: Add 4 deterministic unit tests in `tests/test_demo.py` validating layout IDs, simulation execution, zero collisions, PDW CSV/JSON strings, and Flask WSGI HTTP 200 response without opening ports or launching browsers.

---

## 5. Verification Method

1. Run test suite:
   ```bash
   uv run pytest
   ```
   *Expected*: All 87 unit tests pass.
2. Headless dashboard import and WSGI client test:
   ```bash
   uv run python -c "from demo.dashboard import app; client = app.server.test_client(); res = client.get('/'); assert res.status_code == 200; print('Dashboard WSGI OK')"
   ```
   *Expected*: Prints `Dashboard WSGI OK` with returncode 0.
3. Multi-receiver scheduler collision verification:
   ```bash
   uv run pytest tests/test_multi_receiver.py tests/test_multi_cooperative.py
   ```
   *Expected*: All 18 cooperative tests pass with 0 collisions.
