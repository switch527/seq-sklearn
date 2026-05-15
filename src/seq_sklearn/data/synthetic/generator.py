"""Synthetic data-generating process (F6).

Produces panels of ``(id, time, features, target)`` with a fully specified
DGP so the N1 acceptance thresholds are reproducible byte-for-byte across
re-implementations. Every sampling step draws from one
:class:`numpy.random.Generator` threaded in the F6 order; see
:func:`seq_sklearn.data.synthetic._rng.make_generator`.

F6 leaves a few knobs unpinned. This module fixes them with documented
defaults and exposes them as constructor arguments:

* Per-categorical-column cardinality (``categorical_cardinality``,
  default 4). F6 names ``K_i`` but never pins it; 4 keeps transition
  matrices small while exercising multi-level Markov behaviour.
* The DGP-internal categorical embedding dimension
  (``categorical_embed_dim``, default 3). Used only inside ``phi``; it is
  not the model's embedding size.
* Period grain (``period_grain``, default ``"monthly"``). F6 lists day,
  week, month, quarter, year; monthly matches the canonical
  ``periods_per_entity=(1, 60)`` coverage scenario.
* Target-window emission rule: a row is emitted as a labelled example
  only when a full ``lookback`` window ends at it AND a target row exists
  ``prediction_step`` periods after the window end. Entities shorter than
  ``lookback`` still emit one left-padded window (consistent with F3
  variable-history handling) provided the prediction-step target exists;
  the padded timesteps carry zeros and are flagged downstream by
  :class:`~seq_sklearn.data.tabular_to_sequence.TabularToSequence`'s
  ``padding_mask``. The synthetic generator itself emits one row per
  valid window end with the feature values at that period.

The default panel column names (``id``, ``time`` plus generated feature
names) line up with :class:`~seq_sklearn.config.tabular.TabularToSequenceConfig`
defaults so a generated panel is directly consumable.
"""

import logging
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from seq_sklearn.data.synthetic._rng import make_generator

__all__ = ["SyntheticPanelGenerator"]

logger = logging.getLogger(__name__)

_AR_COEF = 0.7
_W_TEMPORAL_FLOOR = 0.05
_W_TEMPORAL_MIN_LOADED = 3
_W_TEMPORAL_MAX_RETRIES = 5
_W_TEMPORAL_FALLBACK_VALUE = 0.1

TargetKind = Literal["binary", "multiclass", "regression_point", "regression_quantile"]
PeriodGrain = Literal["daily", "weekly", "monthly", "quarterly", "yearly"]

_GRAIN_FREQ: dict[PeriodGrain, str] = {
    "daily": "D",
    "weekly": "W",
    "monthly": "MS",
    "quarterly": "QS",
    "yearly": "YS",
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class SyntheticPanelGenerator:
    """Deterministic synthetic panel generator (F6).

    The DGP is byte-reproducible per ``seed``. Structural parameters
    (transition matrices, projection ``W``, ``w_temporal``, embedding
    tables) are sampled once at the start of :meth:`generate` from the
    single threaded generator and reused across every entity in that
    panel; that is the operational meaning of F6's "frozen by seed at
    construction time".

    The instance carries ``dgp_version = 1``; any change to the sampling
    procedure bumps it (N1 acceptance thresholds pin a version).
    """

    CANONICAL_SEEDS: tuple[int, int, int] = (42, 137, 9999)

    def __init__(
        self,
        *,
        target_kind: TargetKind = "binary",
        num_entities: int = 64,
        periods_per_entity: int | tuple[int, int] | Callable[[np.random.Generator], int] = 24,
        num_static_categorical: int = 2,
        num_static_real: int = 2,
        num_time_varying_real: int = 3,
        num_time_varying_categorical: int = 2,
        num_classes: int = 3,
        class_balance: float = 0.5,
        class_distribution: tuple[float, ...] | None = None,
        target_scale: float = 1.0,
        target_noise: float = 0.1,
        noise_level: float = 0.1,
        signal_strength: float = 0.9,
        lookback: int = 12,
        prediction_step: int = 1,
        categorical_cardinality: int = 4,
        categorical_embed_dim: int = 3,
        period_grain: PeriodGrain = "monthly",
        seed: int | None = None,
        id_col: str = "id",
        time_col: str = "time",
    ) -> None:
        """Configure the generator.

        Raises:
            ValueError: If a numeric knob is out of its documented range
                (negative counts, ``signal_strength`` outside ``[0, 1]``,
                ``num_classes < 2`` for the multiclass kind, a
                ``class_distribution`` whose length mismatches
                ``num_classes`` or does not sum to one).
        """
        if num_entities < 1:
            raise ValueError(f"num_entities must be >= 1, got {num_entities}")
        if not 0.0 <= signal_strength <= 1.0:
            raise ValueError(f"signal_strength must be in [0, 1], got {signal_strength}")
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if prediction_step < 0:
            raise ValueError(f"prediction_step must be >= 0, got {prediction_step}")
        if categorical_cardinality < 1:
            raise ValueError(f"categorical_cardinality must be >= 1, got {categorical_cardinality}")
        if categorical_embed_dim < 1:
            raise ValueError(f"categorical_embed_dim must be >= 1, got {categorical_embed_dim}")
        if target_kind == "multiclass":
            if num_classes < 2:
                raise ValueError(
                    f"multiclass target_kind needs num_classes >= 2, got {num_classes}"
                )
            if class_distribution is not None:
                if len(class_distribution) != num_classes:
                    raise ValueError(
                        f"class_distribution length {len(class_distribution)} != "
                        f"num_classes {num_classes}"
                    )
                if not np.isclose(sum(class_distribution), 1.0):
                    raise ValueError(
                        f"class_distribution must sum to 1.0, got {sum(class_distribution)}"
                    )
        if not 0.0 < class_balance < 1.0:
            raise ValueError(f"class_balance must be in (0, 1), got {class_balance}")

        self.target_kind: TargetKind = target_kind
        self.num_entities = num_entities
        self.periods_per_entity = periods_per_entity
        self.num_static_categorical = num_static_categorical
        self.num_static_real = num_static_real
        self.num_time_varying_real = num_time_varying_real
        self.num_time_varying_categorical = num_time_varying_categorical
        self.num_classes = num_classes
        self.class_balance = class_balance
        self.class_distribution = class_distribution
        self.target_scale = target_scale
        self.target_noise = target_noise
        self.noise_level = noise_level
        self.signal_strength = signal_strength
        self.lookback = lookback
        self.prediction_step = prediction_step
        self.categorical_cardinality = categorical_cardinality
        self.categorical_embed_dim = categorical_embed_dim
        self.period_grain: PeriodGrain = period_grain
        self.seed = seed
        self.id_col = id_col
        self.time_col = time_col
        self.dgp_version = 1

    @property
    def num_target_outputs(self) -> int:
        """Width of the projection target: ``num_classes`` for multiclass, else 1."""
        if self.target_kind == "multiclass":
            return self.num_classes
        return 1

    @property
    def static_categorical_cols(self) -> list[str]:
        return [f"static_cat_{i}" for i in range(self.num_static_categorical)]

    @property
    def static_real_cols(self) -> list[str]:
        return [f"static_real_{i}" for i in range(self.num_static_real)]

    @property
    def time_varying_real_cols(self) -> list[str]:
        return [f"tv_real_{i}" for i in range(self.num_time_varying_real)]

    @property
    def time_varying_categorical_cols(self) -> list[str]:
        return [f"tv_cat_{i}" for i in range(self.num_time_varying_categorical)]

    def _phi_dim(self) -> int:
        return (
            self.num_static_real
            + self.num_static_categorical * self.categorical_embed_dim
            + self.num_time_varying_real
            + self.num_time_varying_categorical * self.categorical_embed_dim
        )

    def _sample_periods(self, rng: np.random.Generator) -> int:
        spec = self.periods_per_entity
        if callable(spec):
            return int(spec(rng))
        if isinstance(spec, tuple):
            low, high = spec
            return int(rng.integers(low, high + 1))
        return int(spec)

    def _build_w_temporal(self, rng: np.random.Generator) -> np.ndarray:
        """Build the load-bearing temporal weight vector (F6 step 6).

        Deterministic per generator: zero entries below the floor,
        renormalize, redraw on too-few loaded entries (advancing the same
        stream), then apply the documented fallback if the retry cap is
        reached. Always returns a vector with at least three loaded
        entries.
        """
        lookback = self.lookback
        if lookback < _W_TEMPORAL_MIN_LOADED:
            raw = rng.dirichlet(np.ones(lookback))
            return raw / raw.sum()

        last_pre_clip = rng.dirichlet(np.ones(lookback))
        post_clip = np.where(last_pre_clip < _W_TEMPORAL_FLOOR, 0.0, last_pre_clip)
        for _ in range(_W_TEMPORAL_MAX_RETRIES):
            if np.count_nonzero(post_clip) >= _W_TEMPORAL_MIN_LOADED:
                return post_clip / post_clip.sum()
            last_pre_clip = rng.dirichlet(np.ones(lookback))
            post_clip = np.where(last_pre_clip < _W_TEMPORAL_FLOOR, 0.0, last_pre_clip)
        if np.count_nonzero(post_clip) >= _W_TEMPORAL_MIN_LOADED:
            return post_clip / post_clip.sum()

        # The retry cap was reached, so post_clip carries fewer than
        # _W_TEMPORAL_MIN_LOADED non-zero entries. F6 then forces the
        # smallest entries of the most recent pre-clip draw to the
        # fallback value and renormalizes the full vector. Ties resolve
        # to the lowest index.
        chosen = sorted(range(lookback), key=lambda i: (last_pre_clip[i], i))[
            :_W_TEMPORAL_MIN_LOADED
        ]
        fallback = last_pre_clip.copy()
        for idx in chosen:
            fallback[idx] = _W_TEMPORAL_FALLBACK_VALUE
        return fallback / fallback.sum()

    def _embed(self, table: np.ndarray, level: int) -> np.ndarray:
        return table[level]

    def generate(self, seed: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate a panel DataFrame and aligned 1-D target array.

        Sampling order follows F6 steps 1-9 exactly, threaded through the
        single generator from ``seed`` (falling back to the constructor
        ``seed``; ``None`` means nondeterministic).

        Returns a ``(panel, y)`` pair. ``panel`` has columns
        ``[id_col, time_col, *feature_cols]``; ``y`` is aligned row-for-row
        to the panel rows that have a valid target window. A ``None`` seed
        (call-time and constructor) yields a fresh nondeterministic stream.
        """
        effective_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng() if effective_seed is None else make_generator(effective_seed)

        card = self.categorical_cardinality
        emb_dim = self.categorical_embed_dim
        n_target = self.num_target_outputs

        static_cat_embed_tables = [
            rng.standard_normal((card, emb_dim)) for _ in range(self.num_static_categorical)
        ]
        tv_cat_embed_tables = [
            rng.standard_normal((card, emb_dim)) for _ in range(self.num_time_varying_categorical)
        ]
        tv_cat_transitions = [
            np.stack([rng.dirichlet(np.ones(card)) for _ in range(card)])
            for _ in range(self.num_time_varying_categorical)
        ]
        phi_dim = self._phi_dim()
        proj_w = rng.standard_normal((n_target, phi_dim))
        w_temporal = self._build_w_temporal(rng)

        class_offsets = self._class_offsets()

        rows: list[dict[str, object]] = []
        targets: list[object] = []
        freq = _GRAIN_FREQ[self.period_grain]

        for entity in range(self.num_entities):
            n_periods = self._sample_periods(rng)
            if n_periods < 1:
                continue

            static_cat_levels = [
                int(rng.integers(0, card)) for _ in range(self.num_static_categorical)
            ]
            static_real_vals = [float(rng.standard_normal()) for _ in range(self.num_static_real)]

            tv_real = np.empty((n_periods, self.num_time_varying_real))
            for col in range(self.num_time_varying_real):
                x_prev = rng.normal(0.0, 1.0 / np.sqrt(1.0 - _AR_COEF**2))
                for t in range(n_periods):
                    x = _AR_COEF * x_prev + rng.standard_normal()
                    tv_real[t, col] = x + rng.normal(0.0, self.noise_level)
                    x_prev = x

            tv_cat = np.empty((n_periods, self.num_time_varying_categorical), dtype=np.int64)
            for col in range(self.num_time_varying_categorical):
                state = int(rng.integers(0, card))
                transition = tv_cat_transitions[col]
                for t in range(n_periods):
                    tv_cat[t, col] = state
                    state = int(rng.choice(card, p=transition[state]))

            timestamps = pd.date_range(start="2015-01-01", periods=n_periods, freq=freq)

            static_cat_embeds = (
                np.concatenate(
                    [
                        self._embed(static_cat_embed_tables[c], static_cat_levels[c])
                        for c in range(self.num_static_categorical)
                    ]
                )
                if self.num_static_categorical
                else np.empty(0)
            )

            for window_end in range(n_periods):
                target_idx = window_end + self.prediction_step
                if target_idx >= n_periods:
                    continue

                window_start = max(0, window_end - self.lookback + 1)
                window_tv_real = tv_real[window_start : window_end + 1]
                pad = self.lookback - window_tv_real.shape[0]
                if pad > 0:
                    window_tv_real = np.vstack(
                        [np.zeros((pad, self.num_time_varying_real)), window_tv_real]
                    )

                weighted_real = (w_temporal[:, None] * window_tv_real).sum(axis=0)

                last_tv_cat_embed = (
                    np.concatenate(
                        [
                            self._embed(tv_cat_embed_tables[c], int(tv_cat[window_end, c]))
                            for c in range(self.num_time_varying_categorical)
                        ]
                    )
                    if self.num_time_varying_categorical
                    else np.empty(0)
                )

                phi = np.concatenate(
                    [
                        np.asarray(static_real_vals, dtype=float),
                        static_cat_embeds,
                        weighted_real,
                        last_tv_cat_embed,
                    ]
                )

                eps = rng.standard_normal(n_target)
                z = self.signal_strength * (proj_w @ phi) + (1.0 - self.signal_strength) * eps
                z = z + class_offsets

                target = self._sample_target(rng, z)

                row: dict[str, object] = {
                    self.id_col: entity,
                    self.time_col: timestamps[window_end],
                }
                for c, name in enumerate(self.static_categorical_cols):
                    row[name] = static_cat_levels[c]
                for c, name in enumerate(self.static_real_cols):
                    row[name] = static_real_vals[c]
                for c, name in enumerate(self.time_varying_real_cols):
                    row[name] = float(tv_real[window_end, c])
                for c, name in enumerate(self.time_varying_categorical_cols):
                    row[name] = int(tv_cat[window_end, c])
                rows.append(row)
                targets.append(target)

        panel = pd.DataFrame(rows)
        if panel.empty:
            ordered_cols = [
                self.id_col,
                self.time_col,
                *self.static_categorical_cols,
                *self.static_real_cols,
                *self.time_varying_real_cols,
                *self.time_varying_categorical_cols,
            ]
            panel = pd.DataFrame({c: [] for c in ordered_cols})
            panel[self.time_col] = panel[self.time_col].astype("datetime64[ns]")
        y_dtype = np.float64 if self.target_kind.startswith("regression") else np.int64
        y = np.asarray(targets, dtype=y_dtype)
        return panel, y

    def _class_offsets(self) -> np.ndarray:
        """Logit / bias offsets implementing class_balance or class_distribution.

        Binary uses the logit of the requested positive rate; multiclass
        uses log of the requested marginal (a softmax bias). Regression
        carries no offset.
        """
        if self.target_kind == "binary":
            p = self.class_balance
            return np.array([np.log(p / (1.0 - p))])
        if self.target_kind == "multiclass":
            if self.class_distribution is None:
                return np.zeros(self.num_classes)
            return np.log(np.asarray(self.class_distribution, dtype=float))
        return np.zeros(1)

    def _sample_target(self, rng: np.random.Generator, z: np.ndarray) -> object:
        if self.target_kind == "binary":
            prob = float(_sigmoid(z)[0])
            return int(rng.random() < prob)
        if self.target_kind == "multiclass":
            probs = _softmax(z)
            return int(rng.choice(self.num_classes, p=probs))
        return float(self.target_scale * z[0] + rng.normal(0.0, self.target_noise))
