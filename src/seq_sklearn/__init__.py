"""seq-sklearn: sklearn-compatible deep sequence models for tabular panel data.

The public API is defined by `__all__` below per architecture A3
(`docs/architecture.md:219-251`) and the requirements stability tiers
(`docs/requirements.md`). Anything not in this list, or reachable only
through an underscore-prefixed module/attribute, is INTERNAL and not
covered by the SemVer stability guarantee.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

# `__version__` MUST be bound before the deep re-exports below: some
# submodules (e.g. `seq_sklearn.serialization`) do
# `from seq_sklearn import __version__` at their own import time, and
# if the deep imports run first they would trigger a circular-import
# error here while this module is still initializing.
try:
    __version__ = _pkg_version("seq-sklearn")
except _PackageNotFoundError:  # pragma: no cover - source-checkout w/o install
    __version__ = "0.0.0+unknown"

from seq_sklearn.config.tabular import TabularToSequenceConfig
from seq_sklearn.config.tft import TFTConfig
from seq_sklearn.data.tabular_to_sequence import TabularToSequence
from seq_sklearn.errors import (
    ConfigError,
    DataContractError,
    NotFittedError,
    PredictionError,
    SeqSklearnError,
    TrainingError,
)
from seq_sklearn.hardware import HardwareTier, detect
from seq_sklearn.inference.attention import AttentionOutput, RegressionAttentionOutput
from seq_sklearn.model_selection.split import EntityTimeSeriesSplit
from seq_sklearn.models.transformer.tft.classifier import TFTClassifier
from seq_sklearn.models.transformer.tft.regressor import TFTRegressor
from seq_sklearn.tuning.pruning import optuna_trial_guard
from seq_sklearn.tuning.suggest_params import suggest_params

__all__ = [
    "AttentionOutput",
    "ConfigError",
    "DataContractError",
    "EntityTimeSeriesSplit",
    "HardwareTier",
    "NotFittedError",
    "PredictionError",
    "RegressionAttentionOutput",
    "SeqSklearnError",
    "TFTClassifier",
    "TFTConfig",
    "TFTRegressor",
    "TabularToSequence",
    "TabularToSequenceConfig",
    "TrainingError",
    "detect",
    "optuna_trial_guard",
    "suggest_params",
]  # fmt: skip
