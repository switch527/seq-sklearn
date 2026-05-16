"""Single-`Generator` threading helper (F6 step 1).

F6 mandates that every sampling step in the DGP draw from one
``numpy.random.Generator(PCG64(seed))`` instance, threaded through the
steps in the documented order. Splitting into substreams or
constructing a second ``Generator`` mid-way is forbidden: it breaks the
byte-identical reproducibility contract that the N1 acceptance
thresholds depend on. This module is the only sanctioned place that
constructs the generator, so the seeding policy lives in exactly one
spot.
"""

import numpy as np

__all__ = ["make_generator"]


def make_generator(seed: int) -> np.random.Generator:
    """Construct the canonical DGP generator for ``seed``.

    The same ``seed`` produces the same byte sequence on every call.
    Callers must thread the returned instance through every sampling
    step in the F6 order and must not construct a second generator for
    the same panel.
    """
    return np.random.Generator(np.random.PCG64(seed))
