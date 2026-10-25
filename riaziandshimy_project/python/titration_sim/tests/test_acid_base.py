import math
import numpy as np
from tests._compat import raises, close
from titration_sim.engine.acid_base import *
from titration_sim.engine.equilibrium import weak_acid_cubic_ph, henderson_hasselbalch


def sa(**kw):
    return AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0, **kw)


def test_strong_strong_regions():
    m = sa()
    assert close(ph_at(m, 0), 1.0, 1e-6)
    assert close(ph_at(m, 25.0), 7.0, 1e-6)           # equivalence
    assert close(ph_at(m, 12.5), -math.log10(0.05 * 25 / 37.5 * 1.0 / 1.0 * 1.0) if False else -math.log10((2.5e-3 - 1.25e-3) / 0.0375), 1e-6)
    assert ph_at(m, 40.0) > 12


def test_equivalence_ph_depends_on_temperature():
    m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, temperature_c=50.0)
    assert close(ph_at(m, 25.0), pkw_water(50.0) / 2, 1e-6)
    assert abs(ph_at(m, 25.0) - 7.0) > 0.3


def test_pkw_table():
    assert abs(pkw_water(25) - 13.995) < 0.005
    for t, v in ((0, 14.94), (20, 14.17), (37, 13.62), (50, 13.26), (100, 12.26)):
        assert abs(pkw_water(t) - v) < 0.03, t
    with raises(ValueError):
        pkw_water(500)


def test_weak_acid_initial_is_cubic():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)
    assert close(ph_at(m, 0), weak_acid_cubic_ph(10 ** -4.76, 0.1), 1e-8)


def test_half_equivalence_ph_equals_pka():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)
    assert abs(ph_at(m, 12.5) - 4.76) < 0.01


def test_weak_acid_equivalence_basic():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)
    assert 8.6 < ph_at(m, 25.0) < 8.9


def test_exact_vs_henderson_in_buffer_region():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)
    v = 10.0
    n_a = 0.1 * 25 - 0.1 * v
    n_b = 0.1 * v
    assert abs(ph_at(m, v) - henderson_hasselbalch(4.76, n_b, n_a)) < 0.02


def test_weak_base_titrated_with_acid():
    m = AcidBaseModel.weak_base(0.1, 25.0, 0.1, [9.25], pkw=14.0)
    assert 11.0 < ph_at(m, 0) < 11.3
    assert abs(ph_at(m, 12.5) - 9.25) < 0.01
    assert ph_at(m, 25.0) < 7


def test_weak_base_from_pkb():
    m = AcidBaseModel.weak_base_from_pkb(0.1, 25.0, 0.1, 4.75, pkw=14.0)
    assert close(m.analyte_pka[0], 9.25)


def test_polyprotic_phosphoric_half_equivalences():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [2.15, 7.20, 12.35], pkw=14.0)
    eq = equivalence_volumes(m)
    assert eq == [25.0, 50.0, 75.0]
    assert abs(ph_at(m, 37.5) - 7.20) < 0.05
    assert 4.0 < ph_at(m, 25.0) < 5.0  # amphiprotic H2PO4-: (pKa1+pKa2)/2 = 4.675


def test_hexaprotic_runs():
    m = AcidBaseModel.weak_acid(0.004, 25.0, 0.1, [2, 3, 4, 6, 8, 10], pkw=13.77)
    assert 2.0 < ph_at(m, 0) < 3.0
    assert 10.0 < ph_at(m, 6.0) < 11.5


def test_weak_weak_flat_curve():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], titrant_strength=TitrantStrength.WEAK, titrant_pka=9.25, pkw=14.0)
    rep = check_visual_endpoint(m)
    assert not rep.visual_ok and rep.jump < 1.0 and "weak/weak" in rep.message
    assert abs(ph_at(m, 25.0) - 0.5 * (4.76 + 9.25)) < 0.05      # pH_eq ~ (pKa + pKa(BH+))/2


def test_feasibility_threshold_depends_on_concentration():
    # The 'Ka < 1e-10' rule of the source is NOT the criterion: at 0.1 M the jump is already < 2 at pKa ~ 4.8
    assert check_visual_endpoint(AcidBaseModel.weak_acid(0.1, 25, 0.1, [3.0], pkw=14.0)).visual_ok
    assert not check_visual_endpoint(AcidBaseModel.weak_acid(0.1, 25, 0.1, [7.0], pkw=14.0)).visual_ok
    assert not check_visual_endpoint(AcidBaseModel.weak_acid(0.001, 25, 0.001, [5.0], pkw=14.0)).visual_ok


def test_engine_is_pure_and_matches_direct_call():
    eng = AcidBaseEngine()
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)
    s0 = eng.initial_state(m)
    s1 = eng.titrate(m, s0, 10.0)
    s1b = eng.titrate(m, s0, 10.0)
    assert s1 == s1b and s0.v_titrant_ml == 0.0
    s2 = eng.titrate(m, s1, 5.0)
    assert close(s2.ph, ph_at(m, 15.0), 1e-12) and close(s2.volume_ml, 40.0)
    assert close(s2.species["H+"] * s2.species["OH-"], m.kw, 1e-9)


def test_engine_errors():
    eng = AcidBaseEngine()
    m = sa()
    with raises(ValueError):
        eng.titrate(m, eng.initial_state(m), -1.0)
    with raises(ValueError):
        ph_at(m, -0.1)
    with raises(TypeError):
        eng.initial_state(TitrationModelDummy())


from titration_sim.engine.base import TitrationModel
class TitrationModelDummy(TitrationModel):
    pass


def test_model_validation():
    with raises(ValueError):
        AcidBaseModel.strong_acid(-0.1, 25, 0.1)
    with raises(ValueError):
        AcidBaseModel(AnalyteKind.WEAK_ACID, 0.1, 25, 0.1)
    with raises(ValueError):
        AcidBaseModel(AnalyteKind.STRONG_ACID, 0.1, 25, 0.1, (4.0,))
    with raises(ValueError):
        AcidBaseModel.weak_acid(0.1, 25, 0.1, [7.0, 2.0])
    with raises(ValueError):
        AcidBaseModel(AnalyteKind.STRONG_ACID, 0.1, 25, 0.1, titrant_strength=TitrantStrength.WEAK)


def test_temperature_effects():
    assert close(pka_at_temperature(4.76, 0.0, 50), 4.76)
    assert pka_at_temperature(4.76, -0.4, 50) > 4.76
    m = AcidBaseModel.weak_acid(0.1, 25, 0.1, [4.76], delta_h_kj_mol=-0.4, temperature_c=50.0)
    assert close(m._pka_t(4.76), pka_at_temperature(4.76, -0.4, 50.0))
    assert with_temperature(sa(), 40).temperature_c == 40
    m1 = AcidBaseModel.strong_acid(0.1, 25, 0.1, temperature_c=40.0, thermal_volume_correction=True)
    assert m1.volume_factor > 1.0


def test_curve_adaptive_is_nonuniform_and_refined_near_eq():
    m = sa()
    v, p = generate_curve(m, 30.0, 61)
    assert len(v) > 61 and np.all(np.diff(v) > 0)
    assert np.min(np.diff(v)) < 0.5 * (30 / 60)
    assert np.max(np.abs(np.diff(p))) < 0.06
    u, q = generate_curve(m, 30.0, 61, adaptive=False)
    assert len(u) == 61
    with raises(ValueError):
        generate_curve(m, 0.0)


def test_pka_from_pkb():
    assert close(pka_from_pkb(4.75, 14.0), 9.25)
