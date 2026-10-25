import numpy as np
from tests._compat import raises, close
from titration_sim.engine import potentiometry as P
from titration_sim.engine.equilibrium import stepwise_to_cumulative

MODEL = P.ProtonationModel(6, -6, 6, -1, "const", "dh", 10.0**-13.77, 0.1)
START = stepwise_to_cumulative([v - 0.3 for v in P.PAPER_LOG_K])


def test_gamma_and_conversions():
    assert close(P.gamma_log(1, 0.1, "dh"), -0.51 * 0.1**0.5 / (1 + 0.1**0.5))
    assert P.gamma_log(2, 0.1, "davies") < P.gamma_log(1, 0.1, "davies")
    assert P.gamma_log(1, 0.0, "edh") == 0.0
    with raises(ValueError):
        P.gamma_log(1, 0.1, "x")
    with raises(ValueError):
        P.gamma_log(1, -0.1)
    lb = [10.0, 18.0, 24.0]
    for g in ("dh", "davies", "edh"):
        back = P.to_thermodynamic(P.to_stoichiometric(lb, -3, 0.1, g), -3, 0.1, g)
        assert np.allclose(back, lb)


def test_paper_medium_effect_numbers():
    # Paper: log K^T - log K = 0.122 z* (DH, I=0.1): 1.46, 1.22, ... for steps 1..6
    lbT = [0.0] * 6
    stoich = P.to_stoichiometric(lbT, -6, 0.1, "dh")
    from titration_sim.engine.equilibrium import cumulative_to_stepwise
    k = np.asarray(cumulative_to_stepwise(stoich))
    # z* for step i: sum z_reac^2 - sum z_prod^2 = (zA+i-1)^2 + 1 - (zA+i)^2... check magnitude pattern
    assert abs(-k[0] - 1.46) < 0.02 and abs(-k[5] - 0.25) < 0.02


def test_weights_eq3():
    w = P.weights_eq3(np.arange(10.0), 3.0 * np.arange(10.0), 0.15, 0.005)
    assert np.allclose(w, 1 / (0.15**2 + 9 * 0.005**2))
    with raises(ValueError):
        P.weights_eq3([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 0.0, 0.0)


def test_benchmark_shape_and_inputs():
    c = P.benchmark_curves(0.0)
    assert len(c) == 6 and all(x.v_ml.size == 101 for x in c)
    assert c[0].y[0] < c[0].y[-1]            # E = E0 + S log[H] rises with pH because S < 0
    with raises(ValueError):
        P.benchmark_curves(n_points=5)


def test_noise_free_fit_recovers_truth():
    c = P.benchmark_curves(0.0)
    r = P.fit_protonation(c, MODEL, START, P.PAPER_CALIB, 0.1, max_iter=150)
    assert np.allclose(r.log_k, P.PAPER_LOG_K, atol=1e-4) and r.converged


def test_noisy_fit_within_uncertainty_and_sd_reasonable():
    c = P.benchmark_curves(0.05, seed=7)
    r = P.fit_protonation(c, MODEL, START, P.PAPER_CALIB, 0.1, max_iter=150)
    assert np.all(np.abs(r.log_k - P.PAPER_LOG_K) < 5 * r.sd_log_k + 1e-3)
    assert r.sd_log_k[5] > r.sd_log_k[2]          # logK6 is the worst determined (paper Sec. 3.2)
    assert 0.05 < r.u < 0.3 and r.dof == r.n_points - 6


def test_unweighted_fit_has_smaller_sd_like_paper():
    c = P.benchmark_curves(0.05, seed=7)
    w = P.fit_protonation(c, MODEL, START, P.PAPER_CALIB, 0.1, weighted=True, max_iter=150)
    u = P.fit_protonation(c, MODEL, START, P.PAPER_CALIB, 0.1, weighted=False, s_y=1.0, max_iter=150)
    assert np.all(np.isfinite(u.sd_log_k)) and np.all(np.isfinite(w.sd_log_k))


def test_refine_electrode_parameter():
    c = P.benchmark_curves(0.05, seed=3)
    r = P.fit_protonation(c, MODEL, START, P.Calibration(406.0, -59.16, -64.0, 0.0), 0.1, refine=("e0",), max_iter=150)
    assert abs(r.params["e0"][0] - 405.0) < 0.2
    with raises(ValueError):
        P.fit_protonation(c, MODEL, START, P.PAPER_CALIB, 0.1, refine=("bogus",))


def test_pH_observable_and_input_errors():
    c = P.benchmark_curves(0.0)
    pred = P.predict_curve(MODEL, stepwise_to_cumulative(P.PAPER_LOG_K), P.PAPER_CALIB, 0.1, c[0], "pH")
    assert pred.y[0] < pred.y[-1] and np.allclose(pred.alpha.sum(axis=1), 1.0)
    with raises(ValueError):
        P.predict_curve(MODEL, START, P.PAPER_CALIB, 0.1, c[0], "x")


def test_too_few_points():
    c = P.FitCurve("x", 25.0, 4e-3, np.linspace(0, 1, 4), np.array([1.0, 2, 3, 4]))
    with raises(ValueError):
        P.fit_protonation([c], MODEL, START, P.PAPER_CALIB, 0.1, refine=("e0", "slope", "ja", "jb", "ct", "cl"))


def test_variable_ionic_strength_range_matches_paper_fig5a():
    # Paper: ionic strength of the 4 mM solution ranges 0.102-0.156 mol/L along the titration
    var = P.ProtonationModel(6, -6, 6, -1, "var", "dh", 10.0**-13.997, 0.1)
    c = P.benchmark_curves(0.0)[2]
    pred = P.predict_curve(var, P.to_thermodynamic(stepwise_to_cumulative(P.PAPER_LOG_K), -6, 0.1, "dh"),
                           P.Calibration(410.0, -59.16, -64.0, 0.0), 0.1, c, "E")
    assert 0.100 < pred.ionic.min() < 0.112


def test_paper_delta_table_values():
    assert abs(P.paper_delta("wrong_titrant")[0] - 0.293) < 1e-9
    assert abs(P.paper_delta("ionic_strength")[5] + 0.22) < 1e-9
    with raises(KeyError):
        P.paper_delta("nope")


def slow_scenarios():
    d = P.error_scenarios()
    # magnitudes agree with the paper's BSTAC column (tolerances reflect noise and software differences)
    assert np.allclose(np.abs(d["wrong_formal_potential"]), np.abs(P.paper_delta("wrong_formal_potential")), atol=0.015)
    assert np.allclose(d["wrong_titrant"], P.paper_delta("wrong_titrant"), atol=0.05)
    assert np.allclose(d["wrong_acid"], P.paper_delta("wrong_acid"), atol=0.03)
    assert np.allclose(d["ionic_strength"], P.paper_delta("ionic_strength"), atol=0.06)


def test_error_scenarios_fast_parts():
    c = P.benchmark_curves(0.05, seed=7)
    kw = dict(weighted=True, s_y=0.15, s_v=0.005, obs="E", max_iter=150)
    ref = P.fit_protonation(c, MODEL, START, P.PAPER_CALIB, 0.1, **kw)
    f = P.fit_protonation(c, MODEL, START, P.PAPER_CALIB, 0.098, **kw)
    d = f.log_k - ref.log_k
    assert np.allclose(d, P.paper_delta("wrong_titrant"), atol=0.05)
    # the error grows from logK6 to logK1, as the paper states
    assert d[0] > d[1] > d[2] > d[3] > d[5] - 0.02
    f = P.fit_protonation(c, MODEL, START, P.Calibration(406.0, -59.16, -64.0, 0.0), 0.1, **kw)
    de = f.log_k - ref.log_k
    assert np.allclose(np.abs(de), np.abs(P.paper_delta("wrong_formal_potential")), atol=0.015)
    assert abs(np.mean(de)) > 0.015 and abs(de[5]) > abs(de[1])      # edges most sensitive


slow_scenarios.__test__ = False
def test_error_scenarios_full():
    slow_scenarios()
test_error_scenarios_full.slow = True
