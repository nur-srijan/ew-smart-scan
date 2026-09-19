## 2026-09-19T07:52:24Z

You are Worker M3 (C2-ESM Tactical TOC Dashboard Specialist).
Your working directory is: /Users/nursrijan/dev/sih-project/.agents/teamwork_preview_worker_m3
Authoritative Request: /Users/nursrijan/dev/sih-project/.agents/ORIGINAL_REQUEST.md

You MUST read /Users/nursrijan/dev/sih-project/.agents/ORIGINAL_REQUEST.md before starting work.
Also read:
- /Users/nursrijan/dev/sih-project/PROJECT.md
- /Users/nursrijan/dev/sih-project/.agents/teamwork_preview_explorer_survey_3/report.md

Exclusive Write Ownership:
- demo/dashboard.py
- demo/assets/ (optional CSS if needed)

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Mission:
Redesign `demo/dashboard.py` into a professional military-grade C2-ESM (Autonomous Spectrum Surveillance & Telemetry Center) interface using Plotly Dash (backed by Flask WSGI) fulfilling all Requirement R3 acceptance criteria:
1. Fleet Telemetry Matrix:
   - Live telemetry cards for 3 distributed nodes:
     * Node Alpha (UAV-1): Tuner 0 (Phase-Locked Pulse Tracker, Fixed Radars)
     * Node Bravo (UAV-2): Tuners 1 & 2 (Agile FHSS Chaser Pair)
     * Node Charlie (Ground Station): Tuner 3 (Wideband Sentry, Max-AoI Patrol)
   - Display operating mode, operational health status (HEALTHY, DEGRADED, SYNCED), battery/link quality, and assigned tuner frequency allocations in real time.
2. Interactive Multi-Tuner Waterfall:
   - Real-time 2D time-frequency spectrogram showing pulse hits across K=35 sub-bands.
   - 4 distinct color-coded tuner dwell bands overlaid on the spectrogram.
   - Agile emitter hop tracks.
   - Prominent zero tuner collision indicator (e.g., "TUNER COLLISIONS: 0.0% [GUARANTEED]").
3. Electronic Order of Battle (EOB) Threat Table:
   - Live threat identification table displaying: Emitter ID, Type (Fixed, FHSS, Scanning), Center Frequency (GHz), Estimated PRI (µs), AoI (freshness), and Alert Level (CRITICAL, HIGH, MEDIUM, SURVEILLANCE).
   - Styled with military dark-mode aesthetic and color-coded alert badges.
4. PDW Intercept Log & Data Export:
   - Live stream of intercepted Pulse Descriptor Words (timestamp, tuner ID, freq GHz, RSSI dBm, pulse width ns).
   - One-click CSV export button and one-click JSON export button using Dash's `dcc.Download` component.
5. AI vs Legacy Comparison HUD:
   - Prominent metrics banner showing real-time gains in:
     * Interception Ratio (IR) AI vs Legacy (e.g. 91.4% vs 43.8%, +108% gain)
     * Time-to-Intercept (TTI) (e.g. 1.8ms vs 7.4ms, -75% latency)
     * Pulse Interception Throughput (pulses/sec).
6. Operational & Headless Engineering:
   - Ensure the script runs cleanly via `uv run demo/dashboard.py` (and `uv run streamlit run demo/dashboard.py` alias or direct execution).
   - Structure layout and callback functions so they can be tested headlessly via Flask WSGI test client without opening a browser or hanging.

Verification:
- Run headless tests to verify layout generation, callbacks, and export buttons.
- Ensure all 87 existing unit tests continue to pass: `uv run pytest`.

Deliverables:
Write detailed implementation report to /Users/nursrijan/dev/sih-project/.agents/teamwork_preview_worker_m3/report.md and handoff to /Users/nursrijan/dev/sih-project/.agents/teamwork_preview_worker_m3/handoff.md.
