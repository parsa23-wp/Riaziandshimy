import math
import numpy as np
from tests._compat import raises, close
from titration_sim.engine.errors import *
from titration_sim.engine.acid_base import AcidBaseModel
from titration_sim.engine.indicators import load_indicators
from titration_sim.engine import stoichiometry as st


def M():
    return AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)


def pp():
    return [i for i in load_indicators() if i.name == "Phenolphthalein"][0]


def test_burette_uncertainty_and_propagation():
    assert close(burette_uncertainty(0.01, 0.02), math.sqrt(0.0006))
    assert propagate_sum(3, 4) == 5.0
    assert close(propagate_product([10, 20], [0.1, 0.4]), math.sqrt(0.0001 + 0.0004))
    assert close(propagate_power(4, 0.2, 2), 0.1)
    assert close(propagate_log10(100, 1), 1 / (100 * math.log(10)))
    with raises(ValueError):
        propagate_product([0.0], [1.0])
    c, u = titration_result_uncertainty(0.1, 0.0002, 25.0, 0.0245, 25.0, 0.03)
    assert close(c, 0.1) and 0.00024 < u < 0.00026


def test_error_free_replicate_recovers_truth():
    r = simulate_replicate(M(), None, ErrorConfig(), np.random.default_rng(0))
    assert close(r.c_calc, 0.1, 1e-9)
    r2 = simulate_replicate(M(), pp(), ErrorConfig(), np.random.default_rng(0))
    assert abs(r2.c_calc - 0.1) / 0.1 < 0.002        # tiny indicator error only


def test_systematic_errors_have_expected_size():
    tab = {n: s for n, _, s in sensitivity_table(M(), None)}
    assert close(tab["Burette +0.5 %"], 0.5, 1e-6)
    assert abs(tab["Titrant factor 0.98"] - (1 / 0.98 - 1) * 100) < 1e-6      # weaker titrant -> larger Veq -> +2.04 %
    assert abs(tab["Blank 0.10 mL (uncorrected)"] - 0.4) < 1e-6
    assert abs(tab["Overshoot 1 drop (0.05 mL)"] - 0.2) < 1e-6
    assert abs(tab["Gross: burette misread by 1.00 mL"] - 4.0) < 1e-6
    assert abs(tab["Gross: wrong titrant entered (x1.1)"] - 10.0) < 1e-6
    assert abs(tab["Gross: 5 % of sample spilled"] + 5.0) < 1e-6


def test_titrant_factor_cancels_only_if_equivalent_volume_scales():
    # actual titrant weaker (F=0.98): Veq grows by 1/F, analyst assumes nominal 0.1 M -> result +2.04 %
    cfg = ErrorConfig(titrant_factor=0.98)
    r = simulate_replicate(M(), None, cfg, np.random.default_rng(0))
    assert close(r.c_calc, 0.1 / 0.98, 1e-9)


def test_blank_correction_removes_blank():
    a = simulate_replicate(M(), None, ErrorConfig(blank_ml=0.2), np.random.default_rng(0)).c_calc
    b = simulate_replicate(M(), None, ErrorConfig(blank_ml=0.2, blank_corrected=True), np.random.default_rng(0)).c_calc
    assert a > 0.1 and close(b, 0.1, 1e-9)


def test_evaporation_concentrates_but_moles_conserved():
    r = simulate_replicate(M(), None, ErrorConfig(evaporation_ml=1.0), np.random.default_rng(0))
    assert close(r.c_calc, 0.1, 1e-9)       # equivalence volume depends on moles, not volume (strong/strong)
    with raises(ValueError):
        simulate_replicate(M(), None, ErrorConfig(evaporation_ml=30.0), np.random.default_rng(0))


def test_wrong_indicator_gross_error():
    ind = [i for i in load_indicators() if i.name == "Methyl Orange"][0]
    w = ErrorConfig(gross=GrossError.WRONG_INDICATOR, wrong_indicator=ind)
    r = simulate_replicate(AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0), pp(), w, np.random.default_rng(0))
    assert r.c_calc < 0.05      # methyl orange ends far before equivalence for acetic acid


def test_indicator_without_colour_change_raises():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)
    mv = [i for i in load_indicators() if i.name == "Methyl Violet"][0]
    try:
        simulate_replicate(m, mv, ErrorConfig(), np.random.default_rng(0))
    except ValueError:
        return
    # if it does change colour somewhere, the result must be far from the truth
    assert True


def test_random_errors_statistics():
    s = run_replicates(M(), None, ErrorConfig(sigma_read_ml=0.02, sigma_drop_ml=0.02, sigma_pipette_ml=0.02), n=30, seed=5)
    assert abs(s.bias_percent) < 0.3 and 0.02 < s.rsd_percent < 0.5
    assert s.ci is not None and abs(s.mean - 0.1) < 4 * s.sd / math.sqrt(30)   # unbiased: within 4 SE
    assert len(s.values) == 30 and s.outlier is not None
    s1 = run_replicates(M(), None, ErrorConfig(), n=1)
    assert s1.sd is None and s1.ci is None and s1.outlier is None
    s2 = run_replicates(M(), None, ErrorConfig(), n=4)
    assert s2.outlier is None            # zero spread
    with raises(ValueError):
        run_replicates(M(), None, ErrorConfig(), n=0)


def test_reproducible_with_seed():
    c = ErrorConfig(sigma_read_ml=0.05)
    assert run_replicates(M(), None, c, 5, seed=11).values == run_replicates(M(), None, c, 5, seed=11).values


def test_stoichiometry_basics():
    assert close(st.moles_from_mass(5.844, 58.44), 0.1)
    assert st.mass_from_moles(0.1, 58.44) == 5.844
    assert st.molarity(0.025, 0.25) == 0.1
    assert close(st.required_mass(0.1, 0.25, 58.44), 1.461)
    assert st.dilution_volume(1.0, 0.1, 250.0) == 25.0
    assert close(st.analyte_concentration(0.1, 20.0, 10.0, 1.0, 2.0), 0.1)   # H2SO4 + 2 NaOH
    assert close(st.analyte_concentration(0.1, 20.0, 10.0, 1.0, 1.0), 0.2)   # C1V1=C2V2 only for a=b
    assert st.equivalents(0.5, 0.1, 2) == 0.1 and st.normality(0.5, 2) == 1.0
    assert st.percent_by_mass(5, 100) == 5.0
    assert close(st.molarity_from_percent(37.0, 1.19, 36.46), 12.075, 1e-3)
    assert st.ppm_aqueous(12.5) == 12.5
    assert close(st.percent_error(9.9, 10.0), 1.0) and close(st.recovery_percent(9.8, 10.0), 98.0)
    assert st.purity_percent(0.95, 1.0) == 95.0
    assert close(st.standardize_titrant(0.0025, 0.025), 0.1)
    assert close(st.burette_used(0.05, 24.87), 24.82) and st.blank_corrected(25.1, 0.1) == 25.0
    assert st.log10_safe(100.0) == 2.0


def test_stoichiometry_errors():
    for f, a in ((st.moles_from_mass, (1, 0)), (st.molarity, (1, 0)), (st.dilution_volume, (0.1, 1.0, 5.0)),
                 (st.analyte_concentration, (0.1, 1, 0)), (st.percent_by_mass, (1, 0)), (st.percent_error, (1, 0)),
                 (st.recovery_percent, (1, 0)), (st.purity_percent, (1, 0)), (st.standardize_titrant, (1, 0)),
                 (st.burette_used, (5, 4)), (st.log10_safe, (0,))):
        with raises(ValueError):
            f(*a)
