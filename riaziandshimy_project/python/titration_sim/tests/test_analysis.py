import math
import numpy as np
from tests._compat import raises, close
from titration_sim.engine.analysis import *
from titration_sim.engine.acid_base import AcidBaseModel, generate_curve, equivalence_volumes, ph_at


def test_first_derivative_uniform_equals_central_difference():
    x = np.linspace(0, 1, 11)
    y = x**3
    d = first_derivative(x, y)
    cd = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    assert np.allclose(d[1:-1], cd)


def test_first_derivative_nonuniform_exact_for_quadratic():
    x = np.array([0, 0.1, 0.5, 0.6, 1.5, 2.0])
    y = 3 * x**2 + 2 * x
    assert np.allclose(first_derivative(x, y)[1:-1], 6 * x[1:-1] + 2)


def test_second_derivative():
    x = np.array([0, 0.3, 0.4, 1.0, 1.7, 2.0])
    assert np.allclose(second_derivative(x, x**2)[1:-1], 2.0)


def test_derivative_input_checks():
    with raises(ValueError):
        first_derivative([0, 1], [0, 1])
    with raises(ValueError):
        first_derivative([0, 1, 1], [0, 1, 2])
    with raises(ValueError):
        second_derivative([0, 1], [0, 1])


def test_savgol_reduces_noise_and_keeps_polynomials():
    x = np.linspace(0, 10, 101)
    rng = np.random.default_rng(0)
    y = np.sin(x) + rng.normal(0, 0.05, x.size)
    s = savgol_smooth(x, y, 9, 2)
    assert np.std(s - np.sin(x)) < np.std(y - np.sin(x))
    with raises(ValueError):
        savgol_smooth(x, y, 8, 2)
    assert np.allclose(local_slope_poly(x, 2 * x + 1), 2.0)


def _curve(pka=4.76, noise=0.0, seed=0):
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [pka], pkw=14.0)
    v, p = generate_curve(m, 30.0, 121)
    if noise:
        p = p + np.random.default_rng(seed).normal(0, noise, p.size)
    return m, v, p


def test_equivalence_methods_agree_on_exact_curve():
    m, v, p = _curve()
    assert abs(equivalence_first_derivative(v, p) - 25.0) < 0.01
    assert abs(equivalence_second_derivative(v, p) - 25.0) < 0.01
    assert abs(find_equivalence(v, p) - 25.0) < 0.01


def test_equivalence_with_noise():
    m, v, p = _curve(noise=0.01)
    assert abs(find_equivalence(v, p) - 25.0) < 0.1


def test_polyprotic_with_expected_window():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [2.15, 7.20, 12.35], pkw=14.0)
    v, p = generate_curve(m, 90.0, 301)
    e2 = equivalence_first_derivative(v, p, expected=50.0)
    assert abs(e2 - 50.0) < 0.3


def test_gran_strong_acid_exact():
    m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    v = np.linspace(1, 24, 60)
    p = np.array([ph_at(m, x) for x in v])
    g = gran_plot(v, p, 25.0, "strong_acid", 25.0)
    assert abs(g.v_eq - 25.0) < 1e-3 and abs(g.slope + 0.1) < 1e-3 and g.constant is None


def test_gran_weak_acid_corrected_formula_gives_ka():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)
    v = np.linspace(1, 24, 80)
    p = np.array([ph_at(m, x) for x in v])
    g = gran_plot(v, p, 25.0, "weak_acid", 25.0)
    assert abs(g.v_eq - 25.0) < 0.1
    assert abs(-math.log10(g.constant) - 4.76) < 0.05


def test_gran_bases_and_errors():
    m = AcidBaseModel.weak_base(0.1, 25.0, 0.1, [9.25], pkw=14.0)
    v = np.linspace(1, 24, 80)
    p = np.array([ph_at(m, x) for x in v])
    g = gran_plot(v, p, 25.0, "weak_base", 25.0, kw=1e-14)
    assert abs(g.v_eq - 25.0) < 0.15
    with raises(ValueError):
        gran_plot(v, p, 25.0, "nonsense", 25.0)
    with raises(ValueError):
        gran_plot(v[:2], p[:2], 25.0, "weak_acid", 25.0)


def test_combine():
    assert close(combine_equivalence([10, 20], [1, 3]), 17.5)
    with raises(ValueError):
        combine_equivalence([])
    with raises(ValueError):
        combine_equivalence([1.0], [0.0])


def test_basic_statistics():
    x = [24.82, 24.79, 24.84]
    assert close(mean(x), 24.81666666666667)
    assert close(sd([1, 2, 3]), 1.0)
    assert close(rsd_percent([10, 10.2, 9.8]), 2.0, 1e-9)
    assert close(sem([1, 2, 3, 4]), 0.6454972243679028)
    with raises(ValueError):
        sd([1.0])
    with raises(ValueError):
        mean([])


def test_t_quantiles_against_tables():
    tab = {1: 12.706, 2: 4.303, 4: 2.776, 9: 2.262, 19: 2.093, 29: 2.045}
    for df, v in tab.items():
        assert abs(t_quantile(0.975, df) - v) < 2e-3, df
    assert t_quantile(0.5, 5) == 0.0
    assert close(t_cdf(0.0, 3), 0.5)
    with raises(ValueError):
        t_quantile(1.2, 3)


def test_confidence_interval():
    ci = confidence_interval([10.0, 10.2, 9.9, 10.1, 10.0])
    assert abs(ci.t - 2.776) < 2e-3 and ci.lower < ci.mean < ci.upper
    with raises(ValueError):
        confidence_interval([1.0])
    with raises(ValueError):
        confidence_interval([1.0, 2.0], level=1.5)


def test_outlier_tests():
    x = [10.0, 10.1, 10.05, 10.9]
    assert dixon_q_test(x).is_outlier and dixon_q_test(x).index == 3
    assert not dixon_q_test([10.0, 10.1, 10.05, 10.12]).is_outlier
    assert grubbs_test([10.0, 10.1, 10.05, 10.9, 10.02]).index == 3
    assert not grubbs_test([10.0, 10.1, 10.05, 10.12, 10.02]).is_outlier
    with raises(ValueError):
        dixon_q_test([1, 2])
    with raises(ValueError):
        grubbs_test([1.0, 1.0, 1.0])
    with raises(ValueError):
        grubbs_test([1.0, 2.0])
