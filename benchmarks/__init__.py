"""seq-sklearn comparative benchmark suite (Phase B0 scaffold).

This package is the in-repo harness for the comparative benchmark
suite specified at ``docs/benchmark_suite_design.md`` and built
phase-by-phase per ``docs/benchmark_suite_implementation_plan.md``.

It is NOT shipped in the wheel; ``tests/deploy/test_wheel_install.py``
asserts the exclusion. It consumes the library through its public
façade (`seq_sklearn.__all__`); no deep imports into
``seq_sklearn._*``.
"""

__all__ = []  # populated phase-by-phase as adapters and experiments land
