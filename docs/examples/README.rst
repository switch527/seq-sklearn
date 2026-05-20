Examples gallery
================

A curated set of runnable, CI-executed examples. Every script is
small (CPU + ``max_epochs=1``) so the docs build stays inside the
PR-CI budget; the static ``test_every_gallery_script_has_max_epochs_1``
guard enforces it.

The full N1 quickstart with its real accuracy threshold lives at
``examples/quickstart.py`` at the repo root and is mirrored on the
:doc:`landing page <../index>` via ``literalinclude``; the gallery
versions below illustrate specific tasks (multiclass, quantile
regression, attention extraction, imbalanced classes).
