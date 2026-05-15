# Lightning research (2026)

Scope: locked to the seq-sklearn v1 Lightning integration sketch in
`docs/architecture.md` §A7 and the F-series requirements in
`docs/requirements.md`.

## Source citations

1. https://pypi.org/project/lightning/
2. https://pypi.org/project/pytorch-lightning/
3. https://github.com/Lightning-AI/pytorch-lightning/releases
4. https://github.com/Lightning-AI/pytorch-lightning/discussions/16688
5. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html
6. https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
7. https://lightning.ai/docs/pytorch/stable/common/optimization.html
8. https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
9. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
10. https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
11. https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html
12. https://lightning.ai/docs/pytorch/stable/common/early_stopping.html
13. https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html
14. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.utilities.seed.html
15. https://lightning.ai/docs/pytorch/stable/common/trainer.html
16. https://lightning.ai/docs/pytorch/stable/extensions/logging.html
17. https://lightning.ai/docs/pytorch/stable/advanced/speed.html
18. https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
19. https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.ModelHooks.html
20. https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.PyTorchLightningPruningCallback.html
21. https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
22. https://github.com/Lightning-AI/pytorch-lightning/discussions/14318
23. https://github.com/Lightning-AI/pytorch-lightning/issues/20204

## Version pin recommendation

Latest stable as of May 2026 is `lightning==2.6.1`, released
2026-01-30 [3]. Versions `2.6.2` and `2.6.3` were yanked after a 42-minute
PyPI supply-chain compromise on 2026-04-30 and must not be pinned.

Package and import path consolidated in 1.8.0 [4]:

- Install: `pip install lightning` (mono-package, ships
  `lightning.pytorch`, `lightning.fabric`, plus TorchMetrics).
- Import: `import lightning as L` or
  `from lightning.pytorch import LightningModule, Trainer, callbacks`.
- The legacy `pytorch-lightning` distribution still publishes a shim that
  re-exports from `lightning.pytorch`, but new code should not depend on
  it [1][2].

Recommended seq-sklearn pin in `pyproject.toml`:

```
"lightning>=2.6.1,<2.7"
```

This pins above the compromised 2.6.2 and 2.6.3 while leaving room for
2.6.x patch releases. Bump the upper bound when 2.7 ships and we have
re-run the integration suite.

## LightningModule API surface

From the 2.6.1 reference [5][6]:

- `training_step(self, *args, **kwargs) -> Tensor | Mapping[str, Any] | None`.
  Returning a bare `Tensor` is treated as the loss; returning a `dict` with
  key `"loss"` lets you attach extras the callbacks can read.
- `validation_step(self, *args, **kwargs) -> Tensor | Mapping[str, Any] | None`.
  Same shape; the returned tensor is not used for `.backward()`.
- `test_step`, `predict_step`: same signature shape.
- `configure_optimizers(self)`: see next section.
- `self.log(name, value, *, prog_bar=False, logger=None, on_step=None,
  on_epoch=None, reduce_fx='mean', sync_dist=False, batch_size=None,
  rank_zero_only=False)`. The `logger` kwarg is `Optional[bool]`; `None`
  means "use the default for this hook". Pass `logger=False` to keep a
  metric out of the configured logger while still exposing it to
  callbacks via `trainer.callback_metrics`.

Lifecycle order around an epoch boundary, per the docs and discussion
14318 [22]:

```
on_train_epoch_start
  ... training batches ...
on_validation_epoch_start
  ... validation batches ...
on_validation_epoch_end          # runs first
on_train_epoch_end               # runs after
```

Validation runs inside the train epoch's tail, so metrics logged in
`on_validation_epoch_end` are visible when `on_train_epoch_end` and any
`ReduceLROnPlateau` step fire.

`automatic_optimization` is set as an instance attribute in `__init__`,
typically `self.automatic_optimization = False` for manual optimization
[8]. It is exposed via a property on `LightningModule` but the canonical
override site is `__init__`. seq-sklearn v1 stays on automatic
optimization, so we do not touch it.

## configure_optimizers return shape

Legal return forms in 2.6 [5][7]:

1. A single optimizer.
2. A list/tuple of optimizers.
3. A `(optimizers, lr_schedulers)` tuple.
4. A dict with keys `"optimizer"` and optional `"lr_scheduler"`.
5. A list of such dicts (one per optimizer).
6. `None` (no optimizer; valid only with manual optimization).

For `ReduceLROnPlateau`, Lightning requires the `lr_scheduler_config`
dict to set `monitor` to a metric logged via `self.log(...)` [7]. The
canonical shape seq-sklearn will emit:

```python
def configure_optimizers(self):
    optim = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr,
                              weight_decay=self.cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=3
    )
    return {
        "optimizer": optim,
        "lr_scheduler": {
            "scheduler": sched,
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        },
    }
```

`strict=True` makes Lightning raise if `val/loss` is missing on a step,
which we want as a config bug rather than silent no-op.

## Callbacks

`EarlyStopping` [12] supports `monitor`, `mode`, `patience`, `min_delta`,
and `check_finite` (default `True`). With `check_finite=True` it aborts
on NaN/inf in the monitored metric, covering one half of F9.

`ModelCheckpoint` [9][10] saves model state, optimizer state, scheduler
state, callback states, hyperparameters, and global step when
`save_weights_only=False`. The docs page for `ModelCheckpoint` itself
does not enumerate RNG states. RNG handling lives in Lightning's
loop-level checkpoint connector, not in `ModelCheckpoint`; see RNG
section below.

For seq-sklearn's custom callbacks, the relevant hooks per
`ModelHooks` [19] and the Callback reference [18] are:

- `on_after_backward(trainer, pl_module)`: gradient NaN/inf check before
  the optimizer step. Cleaner than `on_train_batch_end` because it fires
  after `loss.backward()` but before `optimizer.step()`.
- `on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)`:
  inspect `outputs["loss"]`, post-step gradient checks, and the
  GradScaler skip count.
- `on_validation_epoch_end`: Optuna pruning report site (see below).
- `on_save_checkpoint(trainer, pl_module, checkpoint)`: inject the RNG
  states we want to preserve.
- `on_load_checkpoint(trainer, pl_module, checkpoint)`: restore them.

## Precision configuration 2026

`Trainer(precision=...)` accepts these literal values in 2.6 [13]:

- `"32-true"` (default), `"64-true"`, `"16-true"`, `"bf16-true"`
- `"16-mixed"`, `"bf16-mixed"`
- `"transformer-engine"`, `"transformer-engine-float16"`

`"bf16-mixed"` remains the canonical interface for bfloat16 AMP. The
plugin classes (`MixedPrecisionPlugin`, `TransformerEnginePrecision`,
`BitsandbytesPrecision`) still exist and can be passed via
`Trainer(plugins=[...])`, but the string form is preferred and
auto-selects the right plugin. There is no `amp_backend` argument any
more; it was removed during the 1.x to 2.x transition and is absent
from the 2.6 Trainer reference [15].

For seq-sklearn we expose `precision` as a string literal in the config
schema and pass it through. We do not instantiate plugin classes
directly.

## Optuna integration

The canonical callback now lives in the separate `optuna-integration`
distribution [20]. Import path in 2026:

```python
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
```

The legacy `from optuna.integration import PyTorchLightningPruningCallback`
re-export still works as a thin shim but emits a `DeprecationWarning`.
The example in `optuna-examples` [21] uses the new path.

The callback hooks `on_validation_end` and calls `trial.report(...)`
followed by `trial.should_prune()`. It raises
`optuna.TrialPruned` to abort the trial, which Lightning surfaces as a
clean fit return.

Constraints we inherit:

- Distributed training requires Lightning >= 1.6 and Optuna RDB storage.
- The `monitor` name must match a metric logged via `self.log(...)`.
- The callback is per-trial; the seq-sklearn HPO wrapper must construct
  a fresh `Trainer` per trial and inject the trial-bound callback.

seq-sklearn does not ship its own pruning callback. The wrapper accepts
an optional list of user callbacks and the Optuna driver appends one
when running HPO.

## RNG state at resume

`L.seed_everything(seed, workers=True)` [14] seeds Python `random`,
NumPy, `torch`, and (when available) `torch.cuda` for the main process
and derives unique per-worker seeds.

What Lightning's checkpoint includes by default is the model state,
optimizer state, scheduler state, callback state, loop state, and
hyperparameters [10][11]. The 2.x loop connector calls an internal
`_collect_rng_states()` / `_set_rng_states()` pair that captures Python
`random`, NumPy, CPU torch, and CUDA RNG for the current device, and
restores them on resume. This is not user-visible API. Issue 20204 [23]
documents a known gap: when only `load_from_checkpoint` is used (without
`trainer.fit(ckpt_path=...)`), the RNG states are not restored, so the
torch global RNG drifts.

Gaps for F5 ("model weights, optimizer state, scheduler state, RNG
state"):

- Multi-GPU CUDA RNG: only the current device's CUDA RNG is captured by
  the built-in path. For seq-sklearn's single-device v1 this is fine;
  any future multi-GPU path needs a custom hook calling
  `torch.cuda.get_rng_state_all()` / `set_rng_state_all(...)`.
- NumPy and Python `random` are covered.
- For belt-and-braces and to insulate from any internal-API drift, ship
  an `RngStateCallback` that uses `on_save_checkpoint` /
  `on_load_checkpoint` to write our own block under
  `checkpoint["seq_sklearn_rng"]`:

  ```python
  {
      "python":    random.getstate(),
      "numpy":     np.random.get_state(),
      "torch":     torch.get_rng_state(),
      "cuda_all":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
  }
  ```

  Restore in `on_load_checkpoint`. This makes the F5 guarantee
  independent of Lightning's internal connector.

## Mixed-precision divergence detection

The GradScaler skip mechanism lives in `torch.amp.GradScaler`: when the
unscaled gradients contain inf/NaN, `scaler.step(optimizer)` returns
without calling `optimizer.step()`. Lightning's
`MixedPrecisionPlugin` owns the scaler and exposes it as
`trainer.precision_plugin.scaler` in 2.6.

The skip count is not directly exported, so seq-sklearn detects skips by
comparing `scaler.get_scale()` across batches: a scale decrease implies
the scaler invoked its backoff path, which means at least one skipped
step. A `GradScalerWatchdog` callback hooked on `on_train_batch_end`
maintains a rolling counter of consecutive scale-decrease events and
raises `TrainingError` after three in a row, satisfying F9.

Lightning ships a `terminate_on_nan` style behavior through
`EarlyStopping(check_finite=True)` for the monitored validation metric.
This is complementary, not a replacement: it fires at epoch granularity
on a logged metric, while F9 demands a per-batch guard on the training
loss and gradients. seq-sklearn ships both:

1. `EarlyStopping(check_finite=True, monitor="val/loss")` for the
   validation-loss safety net.
2. A custom `NaNGuard` callback on `on_after_backward` that scans
   gradients with `torch.isfinite` and aborts with `TrainingError`.
3. `GradScalerWatchdog` on `on_train_batch_end` for the three-skip rule.

## DataLoader defaults 2026

The Lightning speed guide [17] still recommends `num_workers > 0` and
`pin_memory=True` on CUDA. There is no automatic change to prefetch
behavior between 2.4 and 2.6: Lightning consumes whatever the user
returns from `train_dataloader()` unchanged.

The 2025-era guidance adds two settings worth surfacing in
seq-sklearn's config:

- `persistent_workers=True` avoids the per-epoch worker respawn cost,
  which dominates for short epochs on small panel data. Default this to
  `True` when `num_workers > 0`.
- `prefetch_factor` defaults to 2 in PyTorch; bump to 4 only if profiling
  shows GPU starvation. Leave at the PyTorch default.

The requirements' `num_workers=min(4, os.cpu_count())` and
`pin_memory=True if torch.cuda.is_available() else False` remain
correct. Add `persistent_workers=(num_workers > 0)` to the default
DataLoader builder.

## Trainer argument deprecations

2.6 Trainer accepts the F-series arguments we care about [15]:
`max_epochs`, `accelerator`, `devices`, `precision`,
`accumulate_grad_batches`, `gradient_clip_val`,
`gradient_clip_algorithm`, `deterministic`, `benchmark`,
`detect_anomaly`, `logger`, `callbacks`, `enable_checkpointing`,
`enable_progress_bar`.

Removals and renames since 2.0 that we must not regress to:

- `auto_lr_find`, `auto_scale_batch_size`: removed; use the `Tuner`
  class.
- `gpus`, `tpu_cores`, `num_processes`: removed; use `accelerator` and
  `devices`.
- `amp_backend`, `amp_level`: removed; use the `precision` string.
- `weights_summary`: replaced by the `ModelSummary` callback.
- `track_grad_norm`: removed; log gradient norms manually in
  `on_after_backward`.
- `resume_from_checkpoint` argument on Trainer: removed; pass
  `ckpt_path=` to `trainer.fit()`.

`deterministic=True` still calls `torch.use_deterministic_algorithms(True)`
internally [15]. The `"warn"` variant calls it with `warn_only=True`.
seq-sklearn's `seed` config wires `deterministic="warn"` by default and
`True` when the user opts into strict reproducibility, matching the
PyTorch reproducibility recipe.

Default logger is `TensorBoardLogger` if `tensorboard` is importable,
else `CSVLogger`; `logger=False` disables logging entirely [16]. Per
the pass-through policy, seq-sklearn's `Trainer` wrapper does not set
`logger` unless the user passes one, but it does set `logger=False`
when running inside HPO trials to avoid spamming the default TB
directory.

## Decisions implied for seq-sklearn

- Pin `lightning>=2.6.1,<2.7`. Block 2.6.2 and 2.6.3 explicitly in
  release notes.
- Import as `from lightning.pytorch import ...`; never
  `import pytorch_lightning`.
- `configure_optimizers` returns the dict shape shown above with
  `monitor="val/loss"` and `strict=True`.
- Ship three callbacks of our own: `NaNGuard` (on_after_backward),
  `GradScalerWatchdog` (on_train_batch_end, three-skip abort),
  `RngStateCallback` (on_save_checkpoint / on_load_checkpoint).
- Use `EarlyStopping(check_finite=True, monitor="val/loss")` as the
  epoch-level safety net, distinct from the per-batch guards above.
- Use `Trainer(precision="bf16-mixed")` as the string form; never
  instantiate `MixedPrecisionPlugin` directly.
- Optuna pruning: import from `optuna_integration.pytorch_lightning`;
  the HPO driver injects the per-trial callback.
- DataLoader defaults: `num_workers=min(4, os.cpu_count())`,
  `pin_memory=torch.cuda.is_available()`,
  `persistent_workers=(num_workers > 0)`, leave `prefetch_factor` at the
  PyTorch default.
- Trainer wrapper exposes: `max_epochs`, `accelerator`, `devices`,
  `precision`, `accumulate_grad_batches`, `gradient_clip_val`,
  `gradient_clip_algorithm`, `deterministic`, `benchmark`,
  `detect_anomaly`. Resume goes through `fit(ckpt_path=...)`, not a
  Trainer constructor arg.
- Logger policy: no default; pass-through. Force `logger=False` in HPO
  trials.
