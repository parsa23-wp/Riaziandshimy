import math
import numpy as np
from tests._compat import raises, close
from titration_sim.engine.equilibrium import *


def test_brent_root():
    assert close(brent(lambda x: x * x - 2, 0, 2), math.sqrt(2), 1e-10)


def test_brent_unbracketed():
    with raises(EquilibriumError):
        brent(lambda x: x * x + 1, 0, 2)


def test_strong_acid_exact_formula():
    # [H] = (C + sqrt(C^2 + 4Kw))/2 (second answer 2.2), also near 1e-7
    for c in (1e-1, 1e-3, 1e-7, 1e-8):
        h = (c + math.sqrt(c * c + 4e-14)) / 2
        assert close(-solve_log_h(c, [], 1e-14), -math.log10(h), 1e-9)


def test_strong_acid_1e8_is_below_7():
    # naive pH = 8 is wrong; correct value is 6.9788 (acid + water)
    ph = -solve_log_h(1e-8, [], 1e-14)
    assert 6.97 < ph < 6.99


def test_weak_acid_matches_cubic():
    for ka, c in ((1.8e-5, 0.1), (1e-3, 0.01), (1e-9, 0.1), (1e-7, 1e-4)):
        lig = [Ligand(c, (-math.log10(ka),))]
        got = -solve_log_h(c, lig, 1e-14)
        assert close(got, weak_acid_cubic_ph(ka, c), 1e-8)


def test_diprotic_alpha_sums_to_one():
    a = species_fractions(-4.0, pka_to_log_beta([2.0, 7.0]))
    assert close(a.sum(), 1.0)


def test_pka_to_log_beta_values_and_errors():
    assert pka_to_log_beta([2.0, 7.0]) == (7.0, 9.0)
    with raises(EquilibriumError):
        pka_to_log_beta([7.0, 2.0])
    with raises(EquilibriumError):
        pka_to_log_beta([])


def test_stepwise_cumulative_roundtrip():
    k = (10.0, 8.0, 6.0)
    assert np.allclose(cumulative_to_stepwise(stepwise_to_cumulative(k)), k)


def test_kw_must_be_positive():
    with raises(EquilibriumError):
        solve_log_h(0.1, [], 0.0)


def test_approximation_error_reported():
    ph, err = weak_acid_ph_approx(1.8e-5, 0.1)
    assert abs(err) < 0.01
    ph2, err2 = weak_acid_ph_approx(1e-3, 1e-3)  # strongly dissociated: approximation poor
    assert abs(err2) > 0.1


def test_henderson():
    assert henderson_hasselbalch(4.76, 1, 1) == 4.76
    with raises(EquilibriumError):
        henderson_hasselbalch(4.76, 0, 1)


def test_hexaprotic_balance_monotone():
    lig = [Ligand(4e-3, stepwise_to_cumulative([10, 8, 6, 4, 3, 2]))]
    xs = np.linspace(-12, 0, 50)
    f = [proton_balance(x, 0.0, lig, 10 ** -13.77) for x in xs]
    assert all(f[i] < f[i + 1] for i in range(len(f) - 1))
