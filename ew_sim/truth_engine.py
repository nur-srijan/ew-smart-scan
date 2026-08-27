"""
ew_sim/truth_engine.py
======================
Builds and stores the ground-truth 2D binary spectrum occupancy matrix:

    S[k, t]  ∈  {0, 1}
    k  = sub-band index  (0 … K-1)
    t  = time-slot index (0 … T-1)

S[k, t] = 1  iff  at least one emitter is actively transmitting in sub-band k
               during time slot t, AND the received power exceeds S_min.

The TruthEngine is the single source of ground truth used by:
  - EWSpectrumEnv  (to generate hit/miss feedback)
  - FoMEvaluator   (to count emitted pulses vs intercepted pulses)
  - Visualization  (spectrogram waterfall)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; switch to TkAgg for live
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ew_sim.emitters import BaseEmitter, make_default_scenario


# ---------------------------------------------------------------------------
# Emitter activity record (used by FoM engine)
# ---------------------------------------------------------------------------

@dataclass
class EmitterActivity:
    """Stores per-emitter pulse activity across all time slots."""
    emitter_id: int
    band_index: int                        # primary / last-known band
    active_slots: list[int] = field(default_factory=list)  # slots where S[k,t]=1

    @property
    def total_pulses(self) -> int:
        return len(self.active_slots)


# ---------------------------------------------------------------------------
# TruthEngine
# ---------------------------------------------------------------------------

class TruthEngine:
    """
    Generates and stores the ground-truth RF environment.

    Parameters
    ----------
    K            : number of frequency sub-bands
    T            : number of scheduling time slots
    dwell_us     : receiver dwell time per slot in microseconds
    switch_us    : LO switching/settling time per slot in microseconds
    f_min_ghz    : lower edge of total surveillance bandwidth (GHz)
    f_max_ghz    : upper edge of total surveillance bandwidth (GHz)
    noise_floor  : probability that a quiet band gives a spurious '1' in S
                   (models hardware glitches / strong out-of-band leakage)
    rng          : numpy random generator (for reproducibility)
    """

    def __init__(
        self,
        K: int = 35,
        T: int = 5000,
        dwell_us: float = 50.0,
        switch_us: float = 5.0,
        f_min_ghz: float = 0.5,
        f_max_ghz: float = 18.0,
        noise_floor: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.K = K
        self.T = T
        self.T_slot = (dwell_us + switch_us) * 1e-6   # seconds
        self.f_min = f_min_ghz
        self.f_max = f_max_ghz
        self.noise_floor = noise_floor
        self.rng = rng if rng is not None else np.random.default_rng(0)

        # Bandwidth per sub-band (GHz)
        self.ibw_ghz = (f_max_ghz - f_min_ghz) / K

        # Sub-band centre frequencies (GHz)
        self.band_centres = np.array([
            f_min_ghz + (k + 0.5) * self.ibw_ghz for k in range(K)
        ])

        self._emitters: list[BaseEmitter] = []

        # Built later
        self.S: Optional[np.ndarray] = None            # shape (K, T)  int8
        self.activity: dict[int, EmitterActivity] = {} # emitter_id → EmitterActivity

    # ------------------------------------------------------------------
    # Emitter registration
    # ------------------------------------------------------------------

    def add_emitter(self, emitter: BaseEmitter) -> None:
        """Register an emitter with the truth engine."""
        self._emitters.append(emitter)
        self.activity[emitter.id] = EmitterActivity(
            emitter_id=emitter.id,
            band_index=emitter.primary_band,
        )

    def add_emitters(self, emitters: list[BaseEmitter]) -> None:
        for e in emitters:
            self.add_emitter(e)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, verbose: bool = True) -> np.ndarray:
        """
        Populate S[K, T] by querying each emitter for every time slot.

        Returns
        -------
        S : np.ndarray of shape (K, T), dtype int8
        """
        self.S = np.zeros((self.K, self.T), dtype=np.int8)

        if verbose:
            print(f"[TruthEngine] Building S[{self.K}, {self.T}] "
                  f"with {len(self._emitters)} emitters …")

        for t in range(self.T):
            t_sec = t * self.T_slot
            for emitter in self._emitters:
                band, active = emitter.state_at(t_sec)
                if band is not None and 0 <= band < self.K and active:
                    self.S[band, t] = 1
                    self.activity[emitter.id].active_slots.append(t)

        # Optional: sprinkle noise floor events
        if self.noise_floor > 0:
            noise_mask = self.rng.random((self.K, self.T)) < self.noise_floor
            self.S = np.clip(self.S + noise_mask.astype(np.int8), 0, 1)

        if verbose:
            occupancy = self.S.mean() * 100
            print(f"[TruthEngine] Done. Global occupancy = {occupancy:.2f}%")
            for eid, act in self.activity.items():
                print(f"  Emitter {eid}: {act.total_pulses} active slots "
                      f"({act.total_pulses / self.T * 100:.1f}% duty)")

        return self.S

    # ------------------------------------------------------------------
    # Query helpers (used by EWSpectrumEnv at runtime)
    # ------------------------------------------------------------------

    def is_active(self, band: int, slot: int) -> bool:
        """Return True if sub-band `band` is occupied at time slot `slot`."""
        if self.S is None:
            raise RuntimeError("Call build() before querying the truth engine.")
        return bool(self.S[band, slot])

    def occupancy_per_band(self) -> np.ndarray:
        """Returns fraction of time each band is occupied. Shape: (K,)."""
        if self.S is None:
            raise RuntimeError("Call build() first.")
        return self.S.mean(axis=1)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_waterfall(
        self,
        t_start: int = 0,
        t_end: Optional[int] = None,
        title: str = "RF Ground Truth — Spectrogram Waterfall",
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot S[K, t_start:t_end] as a coloured spectrogram waterfall.

        Sub-bands (frequency) on the Y-axis, time slots on the X-axis.
        Active transmissions shown in orange-red; quiet bands in dark blue.

        Parameters
        ----------
        t_start, t_end : time slot range to display
        save_path      : if given, save PNG to this path
        show           : call plt.show() (requires interactive backend)
        """
        if self.S is None:
            raise RuntimeError("Call build() before plotting.")

        t_end = t_end or min(self.T, t_start + 500)
        S_window = self.S[:, t_start:t_end]

        fig, axes = plt.subplots(
            2, 1,
            figsize=(14, 7),
            gridspec_kw={"height_ratios": [4, 1]},
        )

        # ── Top panel: waterfall ─────────────────────────────────────────
        ax = axes[0]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "ew", ["#0D1B2A", "#E07B39"]    # dark navy → tactical orange
        )
        im = ax.imshow(
            S_window,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[t_start, t_end, -0.5, self.K - 0.5],
            interpolation="nearest",
        )
        ax.set_ylabel("Sub-band (k)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#1C1C1C")

        # Annotate emitter bands
        colours = plt.cm.tab10.colors
        for i, emitter in enumerate(self._emitters):
            ax.axhline(
                emitter.primary_band,
                color=colours[i % 10],
                linewidth=0.8,
                linestyle="--",
                alpha=0.6,
                label=f"E{emitter.id} ({type(emitter).__name__[:5]})",
            )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

        # Y-axis: show GHz labels
        tick_bands = list(range(0, self.K, max(1, self.K // 7)))
        ax.set_yticks(tick_bands)
        ax.set_yticklabels(
            [f"{self.band_centres[b]:.1f}" for b in tick_bands], fontsize=8
        )
        ax.set_ylabel("Centre Freq (GHz)", fontsize=10)

        fig.colorbar(im, ax=ax, label="Occupied (1) / Quiet (0)", shrink=0.6)

        # ── Bottom panel: total occupancy over time ───────────────────────
        ax2 = axes[1]
        total_active = S_window.sum(axis=0)
        t_axis = np.arange(t_start, t_end)
        ax2.fill_between(t_axis, total_active, alpha=0.7, color="#E07B39")
        ax2.set_xlim(t_start, t_end)
        ax2.set_xlabel("Time slot (t)", fontsize=10)
        ax2.set_ylabel("Active\nbands", fontsize=9)
        ax2.set_ylim(0, self.K)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[TruthEngine] Waterfall saved → {save_path}")

        if show:
            plt.show()

        return fig

    def plot_occupancy_bar(
        self,
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Bar chart of time-averaged occupancy per sub-band."""
        if self.S is None:
            raise RuntimeError("Call build() before plotting.")

        occ = self.occupancy_per_band()
        fig, ax = plt.subplots(figsize=(12, 4))
        colours = ["#E07B39" if o > 0 else "#1C3557" for o in occ]
        ax.bar(range(self.K), occ * 100, color=colours, edgecolor="none")
        ax.set_xlabel("Sub-band (k)", fontsize=11)
        ax.set_ylabel("Occupancy (%)", fontsize=11)
        ax.set_title("Time-Averaged Sub-Band Occupancy", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, self.K - 0.5)
        ax.set_ylim(0, 100)

        # Annotate emitter bands
        for emitter in self._emitters:
            ax.axvline(emitter.primary_band, color="red", linewidth=1.2,
                       linestyle=":", alpha=0.8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[TruthEngine] Occupancy bar saved → {save_path}")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the truth matrix and metadata to a compressed .npz file."""
        if self.S is None:
            raise RuntimeError("Call build() before saving.")
        np.savez_compressed(
            path,
            S=self.S,
            band_centres=self.band_centres,
            K=self.K,
            T=self.T,
            T_slot=self.T_slot,
        )
        print(f"[TruthEngine] Saved → {path}.npz")

    @classmethod
    def load(cls, path: str | Path) -> "TruthEngine":
        """Load a previously saved truth engine (no emitter objects restored)."""
        data = np.load(f"{path}.npz" if not str(path).endswith(".npz") else path)
        engine = cls(
            K=int(data["K"]),
            T=int(data["T"]),
        )
        engine.S = data["S"]
        engine.band_centres = data["band_centres"]
        engine.T_slot = float(data["T_slot"])
        return engine


# ---------------------------------------------------------------------------
# Quick builder helper
# ---------------------------------------------------------------------------

def build_default_truth_engine(
    K: int = 35,
    T: int = 5000,
    dwell_us: float = 50.0,
    switch_us: float = 5.0,
    seed: int = 42,
    verbose: bool = True,
) -> TruthEngine:
    """
    Build a TruthEngine with the canonical 5-emitter scenario and return it
    (already built — ready to query).
    """
    engine = TruthEngine(
        K=K, T=T,
        dwell_us=dwell_us,
        switch_us=switch_us,
        rng=np.random.default_rng(seed),
    )
    emitters = make_default_scenario(K=K, rng=np.random.default_rng(seed + 1))
    engine.add_emitters(emitters)
    engine.build(verbose=verbose)
    return engine
