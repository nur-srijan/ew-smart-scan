"""
eval/generate_ppt_graph.py
==========================
Generates PowerPoint-ready, publication-grade compact graphics for presentations.

Outputs:
    1. ppt_slide_executive_summary.png:
       16:9 Widescreen graphic with high-contrast comparison chart + 3 executive KPI cards.
    2. ppt_compact_comparison.png:
       Ultra-clean standalone comparison bar chart designed to drop into any slide.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Paths
ROOT = Path(__file__).parent.parent
ARTIFACT_DIR = Path("/Users/nursrijan/.gemini/antigravity/brain/59130a49-983c-4a01-a3dd-bf066ff9b3a7")
NOTEBOOKS_DIR = ROOT / "notebooks"
EVAL_DIR = ROOT / "eval"

def generate_ppt_graphs():
    print("Generating PPT-optimized executive slide graphics...")

    policies = [
        "Sequential\n(M=1)",
        "Whittle RMAB\n(M=1)",
        "Recurrent DRL\n(M=1)",
        "Multi-Seq\n(M=4)",
        "Multi-Whittle\n(M=4)",
        "Cooperative AI\n(M=4, Ours)",
    ]
    hits_mean = [21.3, 28.5, 29.6, 86.7, 121.7, 175.3]
    hits_std = [4.9, 8.2, 8.0, 31.5, 50.7, 26.4]
    ir_pct = [2.65, 3.80, 3.72, 10.88, 15.23, 22.17]

    colors = [
        "#64748B",  # Slate Gray
        "#2563EB",  # Science Blue
        "#0D9488",  # Deep Teal
        "#D97706",  # Amber
        "#8B5CF6",  # Vibrant Purple
        "#059669",  # DRDO Emerald Green
    ]

    x = np.arange(len(policies))

    # ──────────────────────────────────────────────────────────────────────────
    # 1. FULL EXECUTIVE SLIDE GRAPHIC (16:9 Widescreen)
    # ──────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16.0, 8.0), facecolor="#FFFFFF")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.38, 1.0], wspace=0.16, top=0.88, bottom=0.12, left=0.07, right=0.96)

    # LEFT: Bar Chart with Clear Grouping & High Contrast
    ax_chart = fig.add_subplot(gs[0])

    # Background card shading for groups
    ax_chart.axvspan(-0.5, 2.45, color="#F8FAFC", alpha=1.0, zorder=0)
    ax_chart.axvspan(2.55, 5.5, color="#ECFDF5", alpha=1.0, zorder=0)

    # Group headers placed well above bars
    ax_chart.text(1.0, 245, "SINGLE RECEIVER (M=1)\nPhysical 1-Band Ceiling", ha="center", va="top", fontsize=11, fontweight="bold", color="#64748B")
    ax_chart.text(4.0, 245, "COOPERATIVE (M=4)\nMulti-Payload Fleet Mesh", ha="center", va="top", fontsize=11, fontweight="bold", color="#065F46")

    # M=1 physical ceiling dashed line (only in single-receiver zone)
    ax_chart.plot([-0.5, 2.45], [30.0, 30.0], color="#EF4444", linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)
    ax_chart.text(2.40, 32.5, "M=1 Ceiling (~30)", color="#DC2626", fontsize=9.5, fontweight="bold", ha="right", zorder=5)

    bar_width = 0.54
    bars = ax_chart.bar(x, hits_mean, yerr=hits_std, capsize=5, width=bar_width, color=colors, edgecolor="#1E293B", linewidth=1.2, zorder=3)

    # Number labels on bars
    for i in range(len(bars)):
        h = hits_mean[i]
        err = hits_std[i]
        ir = ir_pct[i]
        if i < 3:
            ax_chart.text(x[i], h * 0.45, f"{h:.0f}\n({ir:.1f}%)", ha="center", va="center", fontsize=10, fontweight="bold", color="#FFFFFF", zorder=4)
        elif i in (3, 4):
            ax_chart.text(x[i], h + err + 4.0, f"{h:.0f}\n({ir:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0F172A", zorder=4)
        else:
            ax_chart.text(x[i], h + err + 4.0, f"{h:.0f} Hits\n({ir:.1f}% IR)", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#065F46", zorder=4)

    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels(policies, fontsize=10.5, fontweight="bold")
    ax_chart.set_ylabel("Captured Radar Pulses (per 1,000 slots)", fontsize=11.5, fontweight="bold")
    ax_chart.set_ylim(0, 260)
    ax_chart.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
    ax_chart.spines["top"].set_visible(False)
    ax_chart.spines["right"].set_visible(False)

    # RIGHT: Executive KPI Summary Cards
    ax_kpi = fig.add_subplot(gs[1])
    ax_kpi.axis("off")
    ax_kpi.set_xlim(-0.02, 1.02)
    ax_kpi.set_ylim(-0.02, 1.02)

    cards_data = [
        {
            "badge": "8.2x TOTAL CAPTURE GAIN",
            "stat": "+723%",
            "color": "#059669",
            "bg": "#ECFDF5",
            "border": "#10B981",
            "title": "Radar Pulses Intercepted",
            "desc": "Surges from 21.3 to 175.3 pulses.\nBreaks single-tuner saturation against\nmulti-emitter dynamic battlefields."
        },
        {
            "badge": "61% FASTER THREAT ALERT",
            "stat": "237 ms",
            "color": "#2563EB",
            "bg": "#EFF6FF",
            "border": "#3B82F6",
            "title": "Mean Time-to-Intercept (TTI)",
            "desc": "Threat warning latency drops from 1,743 ms\nto 237 ms. Agile radars bracketed in\nsub-50 ms with phase-locked tracking."
        },
        {
            "badge": "0.0% COLLISIONS GUARANTEED",
            "stat": "0.0%",
            "color": "#7C3AED",
            "bg": "#F5F3FF",
            "border": "#8B5CF6",
            "title": "Tuner Redundancy Rate",
            "desc": "Strict disjoint role assignment\nguarantees zero overlapping dwells,\ndedicating 100% bandwidth to active search."
        }
    ]

    card_y_positions = [0.69, 0.35, 0.01]
    card_height = 0.29

    for y_pos, data in zip(card_y_positions, cards_data):
        p_box = FancyBboxPatch(
            (0.01, y_pos), 0.98, card_height,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=data["bg"],
            edgecolor=data["border"],
            linewidth=1.5,
            zorder=2
        )
        ax_kpi.add_patch(p_box)

        ax_kpi.text(0.06, y_pos + card_height * 0.82, f"★ {data['badge']}", fontsize=10, fontweight="bold", color=data["color"], va="center", zorder=3)
        ax_kpi.text(0.06, y_pos + card_height * 0.40, data["stat"], fontsize=24, fontweight="bold", color=data["color"], va="center", zorder=3)
        ax_kpi.text(0.44, y_pos + card_height * 0.74, data["title"], fontsize=12, fontweight="bold", color="#1E293B", va="center", zorder=3)
        ax_kpi.text(0.44, y_pos + card_height * 0.35, data["desc"], fontsize=9.5, color="#475569", va="center", zorder=3)

    plt.suptitle("DRDO EW Smart Scan — Multi-Receiver Cooperative Scheduling Performance\nEmpirical Validation on 20 Dynamic Unseen Battlefields with Concurrent Emitters", fontsize=14, fontweight="bold", y=0.96)

    out_exec_art = ARTIFACT_DIR / "ppt_slide_executive_summary.png"
    out_exec_eval = EVAL_DIR / "ppt_slide_executive_summary.png"
    fig.savefig(out_exec_art, dpi=250)
    fig.savefig(out_exec_eval, dpi=250)
    plt.close(fig)
    print(f"Saved: {out_exec_art}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. STANDALONE COMPACT COMPARISON BAR CHART (PPT Slide Drop-in)
    # ──────────────────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10.2, 5.8), facecolor="#FFFFFF")
    plt.subplots_adjust(top=0.88, bottom=0.14, left=0.09, right=0.96)
    
    # Background shading
    ax2.axvspan(-0.5, 2.45, color="#F8FAFC", alpha=1.0, zorder=0)
    ax2.axvspan(2.55, 5.5, color="#ECFDF5", alpha=1.0, zorder=0)

    # Headers placed high
    ax2.text(1.0, 240, "Single Receiver (M=1)\nPhysical 1-Band Ceiling", ha="center", va="top", fontsize=10.5, fontweight="bold", color="#64748B")
    ax2.text(4.0, 240, "Cooperative Fleet (M=4)\nMulti-Payload Orchestration", ha="center", va="top", fontsize=10.5, fontweight="bold", color="#065F46")

    # M=1 ceiling line (only in single-receiver zone)
    ax2.plot([-0.5, 2.45], [30.0, 30.0], color="#EF4444", linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)
    ax2.text(2.40, 32.5, "M=1 Ceiling (~30)", color="#DC2626", fontsize=9.5, fontweight="bold", ha="right", zorder=5)

    bars2 = ax2.bar(x, hits_mean, yerr=hits_std, capsize=5, width=0.52, color=colors, edgecolor="#1E293B", linewidth=1.2, zorder=3)

    for i in range(len(bars2)):
        h = hits_mean[i]
        err = hits_std[i]
        ir = ir_pct[i]
        if i < 3:
            ax2.text(x[i], h * 0.45, f"{h:.0f}\n({ir:.1f}%)", ha="center", va="center", fontsize=10, fontweight="bold", color="#FFFFFF", zorder=4)
        elif i in (3, 4):
            ax2.text(x[i], h + err + 4.0, f"{h:.0f}\n({ir:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0F172A", zorder=4)
        else:
            # Cooperative Role AI top label
            ax2.text(x[i], h + err + 4.0, f"{h:.0f} Hits\n({ir:.1f}% IR)", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#065F46", zorder=4)

    # Callout badge above the chart title or cleanly positioned
    ax2.text(
        4.85, 120, "★ 8.2x GAIN\n+723% Pulses\n0.0% Collisions",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.35", fc="#059669", ec="#047857", lw=1.2),
        zorder=5
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(policies, fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Captured Radar Pulses (per 1,000 slots)", fontsize=11.5, fontweight="bold")
    ax2.set_title("Multi-Receiver Cooperative Scheduling Gain (M=4 vs M=1)", fontsize=13, fontweight="bold", pad=14)
    ax2.set_ylim(0, 255)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    out_comp_art = ARTIFACT_DIR / "ppt_compact_comparison.png"
    out_comp_eval = EVAL_DIR / "ppt_compact_comparison.png"
    fig2.savefig(out_comp_art, dpi=250)
    fig2.savefig(out_comp_eval, dpi=250)
    plt.close(fig2)
    print(f"Saved: {out_comp_art}")

if __name__ == "__main__":
    generate_ppt_graphs()
