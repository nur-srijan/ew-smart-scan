"""
demo/dashboard.py
=================
Interactive Electronic Warfare Smart Scan Web Dashboard (Dash / Plotly).

Provides a real-time graphical control interface for DRDO / SIH evaluators:
    1. Interactive Simulation Controls (Policy Selection, Emitter Presets, Noise, Dwell Times)
    2. Real-Time High-Resolution Spectrogram Waterfall with Receiver Dwell Overlays
    3. Live Figures of Merit (FoM) Telemetry Cards (Pd, Pfa, Interception Ratio, TTI)
    4. Per-Emitter Interception Breakdown Table
    5. Side-by-Side Policy Benchmarking Tab

Usage:
    uv run demo/dashboard.py
    (Then open http://127.0.0.1:8050 in any browser)
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ew_sim.env import EWSpectrumEnv
from ew_sim.truth_engine import TruthEngine, build_default_truth_engine
from ew_sim.emitters import FixedFrequencyEmitter, FHSSEmitter, ScanningEmitter
from ew_sim.turing_loader import SyntheticTuringGenerator, TuringDatasetAdapter
from schedulers.baselines import SequentialSweep, PseudoRandomSweep, PriorityQueueSweep, UniformRandomSweep
from schedulers.rmab import WhittleIndexScheduler
from schedulers.predictor import HybridPredictiveScheduler
from schedulers.drl_agent import DRLScheduler
from eval.fom import FoMEvaluator

CHECKPOINTS = Path(__file__).parent.parent / "checkpoints"


def create_scenario(preset: str, K: int = 35, T: int = 400, seed: int = 42) -> TruthEngine:
    """Constructs a TruthEngine according to the selected tactical preset."""
    if preset == "turing_synthetic":
        gen = SyntheticTuringGenerator(seed=seed)
        pdws = gen.generate_benchmark_pdws(duration_sec=(T * 1.05e-3), num_emitters=5)
        adapter = TuringDatasetAdapter(K=K, T=T, dwell_us=1000, switch_us=50)
        return adapter.pdws_to_truth_engine(pdws)

    engine = TruthEngine(K=K, T=T, dwell_us=1000, switch_us=50, rng=np.random.default_rng(seed))

    if preset == "dense_agile":
        # 4 FHSS emitters + 1 fixed
        engine.add_emitters([
            FixedFrequencyEmitter(0, band_index=4, pri_sec=5.25e-3, pulse_width=1.05e-3),
            FHSSEmitter(1, hop_bands=[2, 5, 8, 12, 16], hop_interval=8.4e-3, pri_sec=2.1e-3, pulse_width=1.05e-3, burst_size=3),
            FHSSEmitter(2, hop_bands=[10, 14, 20, 26, 30], hop_interval=6.3e-3, pri_sec=2.1e-3, pulse_width=1.05e-3, burst_size=2),
            FHSSEmitter(3, hop_bands=[18, 22, 25, 29, 33], hop_interval=10.5e-3, pri_sec=3.15e-3, pulse_width=1.05e-3, burst_size=3),
        ])
    elif preset == "fast_scanning":
        # 2 scanning radars + 2 fixed
        engine.add_emitters([
            FixedFrequencyEmitter(0, band_index=4, pri_sec=5.25e-3, pulse_width=1.05e-3),
            FixedFrequencyEmitter(1, band_index=18, pri_sec=10.5e-3, pulse_width=1.05e-3),
            ScanningEmitter(2, band_index=11, T_scan_sec=1.05, beamwidth_deg=12.0, pri_sec=2.1e-3, pulse_width=1.05e-3, initial_angle=0.0),
            ScanningEmitter(3, band_index=26, T_scan_sec=1.50, beamwidth_deg=10.0, pri_sec=2.1e-3, pulse_width=1.05e-3, initial_angle=45.0),
        ])
    else:
        # Standard Mixed Preset
        engine.add_emitters([
            FixedFrequencyEmitter(0, band_index=4, pri_sec=5.25e-3, pulse_width=1.05e-3),
            FixedFrequencyEmitter(1, band_index=18, pri_sec=10.5e-3, pulse_width=1.05e-3, pri_jitter=0.05),
            FHSSEmitter(2, hop_bands=[7, 10, 14, 20, 26], hop_interval=10.5e-3, pri_sec=3.15e-3, pulse_width=1.05e-3, burst_size=3),
            FHSSEmitter(3, hop_bands=[2, 5, 12, 19, 29, 33], hop_interval=6.3e-3, pri_sec=2.1e-3, pulse_width=1.05e-3, burst_size=2),
            ScanningEmitter(4, band_index=11, T_scan_sec=2.1, beamwidth_deg=10.0, pri_sec=2.1e-3, pulse_width=1.05e-3, initial_angle=0.0, gain_threshold=0.3),
        ])

    engine.build(verbose=False)
    return engine


def instantiate_scheduler(policy_name: str, K: int, seed: int = 42):
    """Factory creating policy instances."""
    if policy_name == "SequentialSweep":
        return SequentialSweep(K=K)
    elif policy_name == "PseudoRandomSweep":
        return PseudoRandomSweep(K=K, seed=seed)
    elif policy_name == "PriorityQueueSweep":
        w = np.ones(K) * 0.5
        w[4] = 3.0
        w[11] = 4.0
        w[18] = 2.5
        return PriorityQueueSweep(K=K, priority_weights=w, seed=seed)
    elif policy_name == "WhittleIndexRMAB":
        return WhittleIndexScheduler(K=K, seed=seed)
    elif policy_name == "HybridPredictiveRMAB":
        return HybridPredictiveScheduler(K=K, seed=seed)
    elif policy_name == "DRLScheduler-RecurrentPPO":
        sb3_path = CHECKPOINTS / "ppo_recurrent_ew.zip"
        pt_path = CHECKPOINTS / "drl_scheduler.pt"
        model_path = sb3_path if sb3_path.exists() else (pt_path if pt_path.exists() else None)
        return DRLScheduler(K=K, model_path=model_path, seed=seed)
    else:
        return UniformRandomSweep(K=K, seed=seed)


# ---------------------------------------------------------------------------
# Dash Application Layout & Styling
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="DRDO EW Smart Scan Dashboard")

app.layout = html.Div(
    style={"backgroundColor": "#0B1120", "color": "#F8FAFC", "fontFamily": "Segoe UI, Arial, sans-serif", "padding": "20px"},
    children=[
        # Header
        html.Div(
            style={"borderBottom": "2px solid #1E293B", "paddingBottom": "15px", "marginBottom": "20px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"},
            children=[
                html.Div([
                    html.H1("⚡ DRDO ELECTRONIC WARFARE — SMART SCAN SCHEDULER", style={"fontSize": "22px", "fontWeight": "bold", "color": "#38BDF8", "margin": 0}),
                    html.P("Autonomous Machine Learning & RMAB Spectrum Surveillance Dashboard · SIH 2026", style={"fontSize": "13px", "color": "#94A3B8", "margin": "4px 0 0 0"}),
                ]),
                html.Div([
                    html.Span("SURVEILLANCE BAND: 0.5 - 18 GHz", style={"backgroundColor": "#1E293B", "padding": "6px 12px", "borderRadius": "6px", "fontSize": "12px", "fontWeight": "bold", "marginRight": "8px", "color": "#38BDF8"}),
                    html.Span("IBW: 500 MHz", style={"backgroundColor": "#1E293B", "padding": "6px 12px", "borderRadius": "6px", "fontSize": "12px", "fontWeight": "bold", "color": "#22C55E"}),
                ]),
            ]
        ),

        # Control Panel & Scenario Configuration
        html.Div(
            style={"backgroundColor": "#1E293B", "padding": "16px", "borderRadius": "8px", "marginBottom": "20px", "display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "15px", "alignItems": "end"},
            children=[
                html.Div([
                    html.Label("Scan Strategy / Policy", style={"fontSize": "12px", "fontWeight": "bold", "color": "#94A3B8"}),
                    dcc.Dropdown(
                        id="policy-dropdown",
                        options=[
                            {"label": "🤖 Recurrent DRL Agent (PPO-LSTM)", "value": "DRLScheduler-RecurrentPPO"},
                            {"label": "⚡ Whittle Index RMAB (Restless Bandit)", "value": "WhittleIndexRMAB"},
                            {"label": "🎯 Hybrid Predictive RMAB", "value": "HybridPredictiveRMAB"},
                            {"label": "📋 Priority Queue (Static EDB)", "value": "PriorityQueueSweep"},
                            {"label": "🎲 Pseudo-Random Permutation", "value": "PseudoRandomSweep"},
                            {"label": "🔄 Sequential Sweep (Legacy Open-Loop)", "value": "SequentialSweep"},
                        ],
                        value="DRLScheduler-RecurrentPPO",
                        style={"color": "#000"},
                    ),
                ]),
                html.Div([
                    html.Label("Tactical Emitter Scenario", style={"fontSize": "12px", "fontWeight": "bold", "color": "#94A3B8"}),
                    dcc.Dropdown(
                        id="scenario-dropdown",
                        options=[
                            {"label": "Standard Mixed (Fixed + FHSS + Rotating)", "value": "standard_mixed"},
                            {"label": "Dense Frequency Hopping (FHSS Heavy)", "value": "dense_agile"},
                            {"label": "Dual Rotating Surveillance Radars", "value": "fast_scanning"},
                            {"label": "Alan Turing Synthetic Radar Dataset Preset", "value": "turing_synthetic"},
                        ],
                        value="standard_mixed",
                        style={"color": "#000"},
                    ),
                ]),
                html.Div([
                    html.Label("Episode Horizon (Time Slots)", style={"fontSize": "12px", "fontWeight": "bold", "color": "#94A3B8"}),
                    dcc.Slider(id="time-slider", min=200, max=800, step=100, value=400, marks={200: "200", 400: "400", 600: "600", 800: "800"}),
                ]),
                html.Div([
                    html.Button("▶ RUN TACTICAL SCAN", id="run-btn", n_clicks=0, style={"backgroundColor": "#0284C7", "color": "#FFF", "border": "none", "padding": "12px 20px", "borderRadius": "6px", "fontWeight": "bold", "cursor": "pointer", "width": "100%"}),
                ]),
            ]
        ),

        # KPI Metrics Row
        html.Div(id="kpi-cards", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "15px", "marginBottom": "20px"}),

        # Visualizations (Spectrogram Waterfall + Telemetry Curves)
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "20px", "marginBottom": "20px"},
            children=[
                html.Div([
                    html.H3("LIVE RF BATTLEGROUND & RECEIVER DWELL OVERLAY", style={"fontSize": "14px", "fontWeight": "bold", "color": "#38BDF8", "marginBottom": "10px"}),
                    dcc.Graph(id="spectrogram-graph", style={"height": "480px"}),
                ], style={"backgroundColor": "#1E293B", "padding": "15px", "borderRadius": "8px"}),

                html.Div([
                    html.H3("PERFORMANCE PROGRESSION TELEMETRY", style={"fontSize": "14px", "fontWeight": "bold", "color": "#38BDF8", "marginBottom": "10px"}),
                    dcc.Graph(id="telemetry-graph", style={"height": "480px"}),
                ], style={"backgroundColor": "#1E293B", "padding": "15px", "borderRadius": "8px"}),
            ]
        ),

        # Emitter Breakdown Table
        html.Div(
            style={"backgroundColor": "#1E293B", "padding": "15px", "borderRadius": "8px"},
            children=[
                html.H3("PER-EMITTER INTERCEPTION & TRACKING BREAKDOWN", style={"fontSize": "14px", "fontWeight": "bold", "color": "#38BDF8", "marginBottom": "10px"}),
                html.Div(id="emitter-table-container"),
            ]
        ),
    ]
)


# ---------------------------------------------------------------------------
# Dashboard Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    [
        Output("kpi-cards", "children"),
        Output("spectrogram-graph", "figure"),
        Output("telemetry-graph", "figure"),
        Output("emitter-table-container", "children"),
    ],
    [Input("run-btn", "n_clicks")],
    [
        State("policy-dropdown", "value"),
        State("scenario-dropdown", "value"),
        State("time-slider", "value"),
    ],
)
def update_dashboard(n_clicks, policy_name, scenario_preset, T_slots):
    K = 35
    seed = 42 + (n_clicks or 0)

    # 1. Build scenario & policy
    truth = create_scenario(scenario_preset, K=K, T=T_slots, seed=seed)
    scheduler = instantiate_scheduler(policy_name, K=K, seed=seed)

    # 2. Run simulation
    env = EWSpectrumEnv(truth_engine=truth, K=K, T=T_slots, Pd=0.95, Pfa=1e-4, seed=seed)
    scheduler.reset(seed=seed)
    obs, info = env.reset(seed=seed)

    actions, hits, rewards, dwell_types = [], [], [], []

    for t in range(T_slots):
        act = scheduler.select_band(obs, info)
        next_obs, rew, term, trunc, next_info = env.step(act)

        is_active = truth.is_active(act, t)
        hit_obs = bool(rew > 0) or bool(is_active and env._rng.random() < 0.95)
        scheduler.update_feedback(act, hit_obs, next_info)

        if is_active and hit_obs:
            dwell_types.append("hit")
        elif is_active and not hit_obs:
            dwell_types.append("miss")
        elif not is_active and hit_obs:
            dwell_types.append("false_alarm")
        else:
            dwell_types.append("quiet")

        actions.append(act)
        hits.append(hit_obs)
        rewards.append(rew)

        obs = next_obs
        info = next_info

    evaluator = FoMEvaluator(truth)
    report = evaluator.evaluate_trajectory(policy_name, actions, hits, rewards)

    # ── KPI Cards ──────────────────────────────────────────────────────────
    ir_val = report.overall_interception_ratio * 100.0
    ir_color = "#22C55E" if ir_val > 20 else ("#F59E0B" if ir_val > 5 else "#EF4444")

    kpis = [
        html.Div([
            html.P("INTERCEPTION RATIO", style={"fontSize": "11px", "color": "#94A3B8", "margin": 0, "fontWeight": "bold"}),
            html.H2(f"{ir_val:.1f}%", style={"fontSize": "26px", "color": ir_color, "margin": "4px 0 0 0"}),
            html.Span(f"{report.total_hits} Pulses Captured", style={"fontSize": "11px", "color": "#CBD5E1"}),
        ], style={"backgroundColor": "#0F172A", "padding": "12px 16px", "borderRadius": "6px", "borderLeft": f"4px solid {ir_color}"}),

        html.Div([
            html.P("MEAN TIME-TO-INTERCEPT", style={"fontSize": "11px", "color": "#94A3B8", "margin": 0, "fontWeight": "bold"}),
            html.H2(f"{report.mean_time_to_intercept_sec:.3f} s", style={"fontSize": "26px", "color": "#38BDF8", "margin": "4px 0 0 0"}),
            html.Span(f"Max: {report.max_time_to_intercept_sec:.2f} s", style={"fontSize": "11px", "color": "#CBD5E1"}),
        ], style={"backgroundColor": "#0F172A", "padding": "12px 16px", "borderRadius": "6px", "borderLeft": "4px solid #38BDF8"}),

        html.Div([
            html.P("EMITTERS DISCOVERED", style={"fontSize": "11px", "color": "#94A3B8", "margin": 0, "fontWeight": "bold"}),
            html.H2(f"{report.emitters_discovered} / {report.total_emitters}", style={"fontSize": "26px", "color": "#A855F7", "margin": "4px 0 0 0"}),
            html.Span(f"{report.discovery_rate*100:.0f}% Spectrum Identified", style={"fontSize": "11px", "color": "#CBD5E1"}),
        ], style={"backgroundColor": "#0F172A", "padding": "12px 16px", "borderRadius": "6px", "borderLeft": "4px solid #A855F7"}),

        html.Div([
            html.P("DETECTION FIDELITY", style={"fontSize": "11px", "color": "#94A3B8", "margin": 0, "fontWeight": "bold"}),
            html.H2(f"{report.empirical_pd*100:.1f}%", style={"fontSize": "26px", "color": "#22C55E", "margin": "4px 0 0 0"}),
            html.Span(f"Pfa: {report.empirical_pfa:.2e}", style={"fontSize": "11px", "color": "#CBD5E1"}),
        ], style={"backgroundColor": "#0F172A", "padding": "12px 16px", "borderRadius": "6px", "borderLeft": "4px solid #22C55E"}),
    ]

    # ── Spectrogram Figure ─────────────────────────────────────────────────
    spec_fig = go.Figure()
    t_axis = list(range(T_slots))

    # Heatmap ground truth
    spec_fig.add_trace(go.Heatmap(
        z=truth.S[:, :T_slots],
        x=t_axis,
        y=list(range(truth.K)),
        colorscale=[[0, "#0F172A"], [1, "#EA580C"]],
        showscale=False,
        hoverinfo="x+y+z",
        opacity=0.65,
        name="Truth Pulses",
    ))

    # Dwell Trajectory & Hit markers
    hit_x = [t for t, d in enumerate(dwell_types) if d == "hit"]
    hit_y = [actions[t] for t in hit_x]
    spec_fig.add_trace(go.Scatter(
        x=hit_x, y=hit_y,
        mode="markers",
        marker=dict(size=8, color="#22C55E", symbol="circle", line=dict(width=1, color="#FFFFFF")),
        name="Intercepted Pulse",
    ))

    quiet_x = [t for t, d in enumerate(dwell_types) if d == "quiet"]
    quiet_y = [actions[t] for t in quiet_x]
    spec_fig.add_trace(go.Scatter(
        x=quiet_x, y=quiet_y,
        mode="markers",
        marker=dict(size=4, color="#38BDF8", opacity=0.4),
        name="Quiet Dwell",
    ))

    spec_fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=20, t=20, b=30),
        xaxis_title="Time Slot Index (t)",
        yaxis_title="Frequency Sub-Band (k)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ── Telemetry Curves ───────────────────────────────────────────────────
    telem_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    
    # Cumulative Hits
    telem_fig.add_trace(go.Scatter(
        x=t_axis, y=report.cumulative_hit_curve,
        line=dict(color="#22C55E", width=2),
        name="Cumulative Hits",
    ), row=1, col=1)

    # Cumulative Discovered Emitters
    telem_fig.add_trace(go.Scatter(
        x=t_axis, y=report.cumulative_discovery_curve,
        line=dict(color="#38BDF8", width=2),
        name="Emitters Discovered",
    ), row=2, col=1)

    telem_fig.update_yaxes(title_text="Pulses", row=1, col=1)
    telem_fig.update_yaxes(title_text="Emitters", row=2, col=1)
    telem_fig.update_xaxes(title_text="Time Slot (t)", row=2, col=1)
    telem_fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=20, t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # ── Emitter Table ──────────────────────────────────────────────────────
    table_data = []
    for eid, em_rep in report.emitter_reports.items():
        table_data.append({
            "Emitter ID": f"Emitter {em_rep.emitter_id}",
            "Type": em_rep.emitter_type,
            "Primary Band": f"Band {em_rep.primary_band} ({truth.band_centres[em_rep.primary_band]:.1f} GHz)",
            "Emitted Pulses": em_rep.total_transmitted_pulses,
            "Captured Pulses": em_rep.intercepted_pulses,
            "Interception Ratio": f"{em_rep.interception_ratio*100:.1f}%",
            "Time-to-Intercept": f"{em_rep.time_to_intercept_sec:.3f} s" if em_rep.discovered else "NOT INTERCEPTED",
            "Status": "✅ TRACKED" if em_rep.discovered else "❌ MISSED",
        })

    em_table = dash_table.DataTable(
        data=table_data,
        columns=[{"name": col, "id": col} for col in table_data[0].keys()] if table_data else [],
        style_header={"backgroundColor": "#0F172A", "color": "#38BDF8", "fontWeight": "bold", "border": "1px solid #334155"},
        style_cell={"backgroundColor": "#1E293B", "color": "#F8FAFC", "padding": "8px 12px", "fontSize": "12px", "border": "1px solid #334155"},
        style_data_conditional=[
            {"if": {"filter_query": '{Status} contains "TRACKED"'}, "color": "#22C55E"},
            {"if": {"filter_query": '{Status} contains "MISSED"'}, "color": "#EF4444"},
        ],
    )

    return kpis, spec_fig, telem_fig, em_table


if __name__ == "__main__":
    print("Starting DRDO EW Smart Scan Live Dashboard on http://127.0.0.1:8050 ...")
    app.run(debug=False, port=8050)
