"""Tests for the single-Generator threading helper (F6 step 1)."""

import numpy as np

from seq_sklearn.data.synthetic._rng import make_generator


def test_same_seed_same_byte_sequence() -> None:
    a = make_generator(42)
    b = make_generator(42)
    draws_a = a.standard_normal(100)
    draws_b = b.standard_normal(100)
    np.testing.assert_array_equal(draws_a, draws_b)


def test_reordering_draws_changes_output() -> None:
    g1 = make_generator(7)
    first_then_second = (g1.standard_normal(5), g1.integers(0, 10, size=5))

    g2 = make_generator(7)
    second_then_first = (g2.integers(0, 10, size=5), g2.standard_normal(5))

    assert not np.array_equal(first_then_second[0], second_then_first[1])


def test_different_seeds_differ() -> None:
    assert not np.array_equal(
        make_generator(1).standard_normal(50),
        make_generator(2).standard_normal(50),
    )
