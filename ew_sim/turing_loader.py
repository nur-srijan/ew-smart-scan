"""
ew_sim/turing_loader.py
=======================
Adapter and Data Loader for the Alan Turing Institute Synthetic Radar Dataset
(huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset).

Transforms continuous Pulse Descriptor Words (PDWs):
    - Time of Arrival (TOA in seconds)
    - Carrier Frequency (f_c in GHz)
    - Pulse Width (PW in seconds)
    - Amplitude (A in dBm or linear scale)
into the discrete 2D Truth Matrix S[K, T] used by the EW Smart Scan environment.

Provides:
    1. TuringDatasetAdapter: Loads local or streamed parquet/csv/json PDW datasets.
    2. SyntheticTuringGenerator: Generates PDW pulse streams matching the exact
       schema of the Alan Turing dataset for offline testing and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import numpy as np

from ew_sim.truth_engine import TruthEngine


@dataclass
class PulseDescriptorWord:
    """Standard Electronic Support Pulse Descriptor Word (PDW)."""
    toa_sec: float           # Time of Arrival
    freq_ghz: float          # Carrier Frequency
    pulse_width_sec: float   # Pulse Width
    amplitude_dbm: float     # Received Peak Power
    emitter_id: int          # Ground truth emitter label


class TuringDatasetAdapter:
    """
    Ingests Alan Turing Radar Dataset PDWs and populates a TruthEngine matrix.
    """

    def __init__(
        self,
        K: int = 35,
        T: int = 2000,
        dwell_us: float = 1000.0,
        switch_us: float = 50.0,
        f_min_ghz: float = 0.5,
        f_max_ghz: float = 18.0,
        sensitivity_dbm: float = -75.0,
    ):
        self.K = K
        self.T = T
        self.T_slot = (dwell_us + switch_us) * 1e-6
        self.f_min = f_min_ghz
        self.f_max = f_max_ghz
        self.ibw_ghz = (f_max_ghz - f_min_ghz) / K
        self.sensitivity_dbm = sensitivity_dbm

    def freq_to_band(self, freq_ghz: float) -> Optional[int]:
        """Maps a continuous frequency in GHz to sub-band index 0..K-1."""
        if freq_ghz < self.f_min or freq_ghz >= self.f_max:
            return None
        return int((freq_ghz - self.f_min) / self.ibw_ghz)

    def toa_to_slot(self, toa_sec: float) -> Optional[int]:
        """Maps continuous Time-of-Arrival in seconds to discrete time slot 0..T-1."""
        slot = int(toa_sec / self.T_slot)
        if 0 <= slot < self.T:
            return slot
        return None

    def pdws_to_truth_engine(
        self,
        pdws: Sequence[PulseDescriptorWord],
    ) -> TruthEngine:
        """
        Converts a list of PDWs into a ready-to-evaluate TruthEngine instance.
        """
        engine = TruthEngine(
            K=self.K,
            T=self.T,
            dwell_us=(self.T_slot * 1e6) - 50.0,
            switch_us=50.0,
            f_min_ghz=self.f_min,
            f_max_ghz=self.f_max,
        )
        engine.S = np.zeros((self.K, self.T), dtype=np.int8)

        for pdw in pdws:
            if pdw.amplitude_dbm < self.sensitivity_dbm:
                continue  # Below receiver sensitivity

            band = self.freq_to_band(pdw.freq_ghz)
            slot = self.toa_to_slot(pdw.toa_sec)

            if band is not None and slot is not None:
                engine.S[band, slot] = 1

        return engine


class SyntheticTuringGenerator:
    """
    Generates synthetic PDWs conforming to the Alan Turing Institute Radar Dataset schema.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_benchmark_pdws(
        self,
        duration_sec: float = 2.1,
        num_emitters: int = 5,
    ) -> list[PulseDescriptorWord]:
        """
        Synthesizes a realistic multi-emitter PDW stream matching the Turing dataset schema.
        """
        pdws: list[PulseDescriptorWord] = []

        # Emitter 1: Fixed-frequency 2.8 GHz, PRI = 5.25 ms
        t = 0.0
        while t < duration_sec:
            pdws.append(PulseDescriptorWord(
                toa_sec=t,
                freq_ghz=2.8,
                pulse_width_sec=2e-6,
                amplitude_dbm=-45.0 + float(self.rng.normal(0, 1.5)),
                emitter_id=1,
            ))
            t += 5.25e-3

        # Emitter 2: Fixed-frequency 9.8 GHz, PRI = 10.5 ms
        t = 0.0
        while t < duration_sec:
            pdws.append(PulseDescriptorWord(
                toa_sec=t,
                freq_ghz=9.8,
                pulse_width_sec=5e-6,
                amplitude_dbm=-50.0 + float(self.rng.normal(0, 1.5)),
                emitter_id=2,
            ))
            t += 10.5e-3 + float(self.rng.uniform(-0.2e-3, 0.2e-3))

        # Emitter 3: Frequency-hopping emitter across 5 agile channels
        hop_freqs = [4.2, 5.8, 7.8, 10.8, 13.8]
        t = 0.0
        while t < duration_sec:
            freq = float(self.rng.choice(hop_freqs))
            for _ in range(3):  # 3 pulses per hop
                pdws.append(PulseDescriptorWord(
                    toa_sec=t,
                    freq_ghz=freq,
                    pulse_width_sec=2e-6,
                    amplitude_dbm=-48.0 + float(self.rng.normal(0, 1.5)),
                    emitter_id=3,
                ))
                t += 3.15e-3
            t += 1.05e-3  # hop gap

        # Emitter 4: Spatially scanning radar at 6.2 GHz, T_scan = 2.1s, Beamwidth = 10 deg
        t = 0.0
        t_scan = 2.1
        tot = (10.0 / 360.0) * t_scan  # ~58 ms mainlobe
        pri = 2.1e-3
        while t < duration_sec:
            # Check if within mainlobe illumination
            phase = t % t_scan
            if phase <= tot:
                # Mainlobe peak power
                amp = -40.0 + float(self.rng.normal(0, 1.0))
                pdws.append(PulseDescriptorWord(
                    toa_sec=t,
                    freq_ghz=6.2,
                    pulse_width_sec=2e-6,
                    amplitude_dbm=amp,
                    emitter_id=4,
                ))
            t += pri

        # Sort PDWs chronologically by TOA
        pdws.sort(key=lambda p: p.toa_sec)
        return pdws
