## Summary

One paragraph explaining what changed and why.

## Phase

Per `docs/implementation_plan.md`:
- [ ] Phase 0  (scaffold)
- [ ] Phase 1  (foundation primitives)
- [ ] Phase 2  (synthetic data + preprocessing)
- [ ] Phase 3  (tensor primitives + serialization)
- [ ] Phase 4a (training plumbing: losses / optimizers / callbacks / sampling)
- [ ] Phase 4b (LightningModule + Trainer + resume_path)
- [ ] Phase 5  (calibration)
- [ ] Phase 6a (BaseSequenceEstimator + smoke skeleton)
- [ ] Phase 6b (family bases + AttentionOutput)
- [ ] Phase 7  (TFT concrete)
- [ ] Phase 8  (public API + Optuna)
- [ ] Phase 9  (check_estimator + acceptance + snapshots)
- [ ] Phase 10 (ONNX + deploy)
- [ ] Phase 11 (perf baselines)
- [ ] Phase 12 (docs + release prep)
- [ ] N/A

## Test plan

- [ ] Local gates pass: `ruff check .`, `ruff format --check .`, `pyright`, `pytest -m "not slow and not perf and not gpu" --cov`
- [ ] Coverage delta is non-negative
- [ ] Touching N1 mandatory tests? Listed below.

## N1 tests added or modified

Cross-reference the test names against `docs/requirements.md` N1 and
`docs/implementation_plan.md` Phase status.

## Snapshot files

- [ ] Not touching `tests/_snapshots/`
- [ ] Touching `tests/_snapshots/`; commit message includes `SNAPSHOT_REVIEWED: <reason>`

## Review

- [ ] `/review` swarm reached consensus
- [ ] `/gemini-final-pass code` ran post-consensus (if non-trivial)
