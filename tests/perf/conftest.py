"""Phase 11 perf fixtures (PG.5).

The determinism fixture is `autouse=False` and requested only by the
`perf`-marked benchmark tests: a session `autouse` fixture would pull
torch at the start of the fast PR suite and break the PC.1a no-torch
boundary for the non-`perf` PG.1/PG.2/PG.3 tests. Both `import torch`
and the `enable_strict_mode` import are INSIDE the fixture body (not
module scope) so importing this conftest to introspect the fixture
does not pull torch (Gemini-C2 / qa R4-I2); PG.3 check (b) pins that.
"""

import pytest


@pytest.fixture(scope="session")
def perf_determinism() -> None:
    """ENABLE then assert strict determinism (PA.2 / Gemini-IMPROVEMENT).

    Enabling-then-asserting both guarantees "determinism ON for every
    perf run" and still fails loudly if `enable_strict_mode` itself
    regresses (the earlier assert-only wording would have crashed every
    run since nothing turned determinism on first).
    """
    import torch

    from seq_sklearn.training._determinism import enable_strict_mode

    enable_strict_mode()
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "perf run requires strict determinism but enable_strict_mode() did not enable it"
        )
