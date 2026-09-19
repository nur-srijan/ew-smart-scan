"""
ew_sim — RF environment simulation package for EW Smart Scan.
"""

from ew_sim.emitters import (
    BaseEmitter,
    FixedFrequencyEmitter,
    FHSSEmitter,
    ScanningEmitter,
    make_default_scenario,
)
from ew_sim.truth_engine import TruthEngine, build_default_truth_engine
from ew_sim.env import EWSpectrumEnv, DynamicSpectrumEnv
from ew_sim.multi_env import MultiReceiverEWSpectrumEnv, DynamicMultiReceiverEnv
from ew_sim.turing_loader import (
    PulseDescriptorWord,
    TuringDatasetAdapter,
    SyntheticTuringGenerator,
)

__all__ = [
    "BaseEmitter",
    "FixedFrequencyEmitter",
    "FHSSEmitter",
    "ScanningEmitter",
    "make_default_scenario",
    "TruthEngine",
    "build_default_truth_engine",
    "EWSpectrumEnv",
    "DynamicSpectrumEnv",
    "MultiReceiverEWSpectrumEnv",
    "DynamicMultiReceiverEnv",
    "PulseDescriptorWord",
    "TuringDatasetAdapter",
    "SyntheticTuringGenerator",
]

