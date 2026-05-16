"""Training plumbing: Lightning callbacks and the F5 builder factories.

Phase 4a ships the standalone pieces (determinism, precision, the
loss / optimizer / scheduler / sampler factories, and the four
callbacks). The ``_LightningModule`` and ``Trainer`` wrapper that
compose them land in Phase 4b.
"""

__all__: list[str] = []
