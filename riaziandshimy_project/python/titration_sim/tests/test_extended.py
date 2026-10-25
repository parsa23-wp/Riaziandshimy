"""Tests for engine/extended.py (non-aqueous, amperometry, thermometry, KF, coulometry, GUM) + acceptance numbers."""
from __future__ import annotations

import math

from titration_sim.engine.acid_base import AcidBaseModel, ph_at
from titration_sim.engine.extended import (
    AmperometricModel, PRIMARY_STANDARDS, Quantity, SOLVENT_PKS, ThermometricModel, amperometric_curve,
    amperometric_current, amperometric_equivalence_volume, biamperometric_current, coulometric_amount_mmol,
    coulometric_efficiency_bias_percent, coulometric_kf_water_ug, coulometric_time_s, coverage_factor,
    deadstop_iodine_curve, gum_budget, kf_titer_mg_per_ml, kf_water_percent, markdown_report,
    nonaqueous_jump, nonaqueous_weak_acid, nonaqueous_weak_base, solvent_pks, standardize_titrant,
    thermometric_curve, thermometric_dt, titration_gum, two_line_endpoint, u_expanded_to_standard,
    u_rectangular, u_triangular,
)
from titration_sim.engine.phase2 import PrecipitationModel, precip_pag

from ._compat import close, raises


# ---------------------------------------------------- acceptance numbers
def test_acceptance_acetic_acid_equivalence_ph() -> None:
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76])
    assert abs(ph_at(m, 25.0) - 8.73) < 0.01


def test_acceptance_agcl_equivalence_pag() -> None:
    m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)
    assert abs(precip_pag(m, 50.0) - 4.87) < 0.01


# ------------------------------------------------------------ non-aqueous
def test_solvent_table_and_errors() -> None:
    assert solvent_pks("acetic_acid") == 14.45
    assert SOLVENT_PKS["methanol"][1] is True and SOLVENT_PKS["dmso"][1] is False
    with raises(ValueError):
        solvent_pks("benzene")


def test_nonaqueous_equivalence_ph_uses_pks() -> None:
    # weak acid pKa 10 in ethanol (pKs 19.5): equivalence pH = (pKa + pKs + log c)/2 with c = 0.05
    m = nonaqueous_weak_acid("ethanol", 0.1, 25.0, 0.1, 10.0)
    expected = 0.5 * (10.0 + 19.5 + math.log10(0.05))
    assert abs(ph_at(m, 25.0) - expected) < 0.02


def test_nonaqueous_makes_weak_acids_titratable() -> None:
    aq = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [10.0])
    na = nonaqueous_weak_acid("acetonitrile", 0.1, 25.0, 0.1, 20.0)
    assert nonaqueous_jump(na) > nonaqueous_jump(aq) + 3.0
    # a base with apparent pKa(BH+) = 6.1 has pKb* = pKs - 6.1 = 8.35: far too weak (jump ~ 0.1, like acetic acid's mirror image)
    weak = nonaqueous_weak_base("acetic_acid", 0.1, 25.0, 0.1, 6.1)
    strong = nonaqueous_weak_base("acetic_acid", 0.1, 25.0, 0.1, 11.0)
    assert weak.pkw == 14.45 and nonaqueous_jump(weak) < 0.5 < 3.0 < nonaqueous_jump(strong)


# ----------------------------------------------------------- amperometry
def test_amperometric_shapes() -> None:
    l_shape = AmperometricModel(0.01, 50.0, 0.1, 1.0, k_analyte=10.0)
    v_shape = AmperometricModel(0.01, 50.0, 0.1, 1.0, k_analyte=10.0, k_titrant=4.0)
    ve = amperometric_equivalence_volume(l_shape)
    assert close(ve, 5.0)
    assert close(amperometric_current(l_shape, 0.0), 100.0)
    assert close(amperometric_current(l_shape, ve), 0.0, 1e-9)
    assert close(amperometric_current(l_shape, 2 * ve), 0.0, 1e-9)
    assert amperometric_current(v_shape, 2 * ve) > 0.0
    v, i = amperometric_curve(v_shape, n=40, dilution_corrected=True)
    assert abs(two_line_endpoint(v, i, ve) - ve) < 1e-6


def test_amperometric_dilution_correction_makes_line_straight() -> None:
    m = AmperometricModel(0.01, 50.0, 0.1, 1.0, k_analyte=10.0)
    pts = [(v, amperometric_current(m, v, True)) for v in (0.5, 1.0, 2.0, 3.0)]
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(pts, pts[1:])]
    assert max(slopes) - min(slopes) < 1e-9
    raw = [amperometric_current(m, v) for v in (0.5, 1.0, 2.0, 3.0)]
    assert not close(raw[1] - raw[0], (raw[2] - raw[1]) / 2.0, 1e-6)


def test_amperometric_errors() -> None:
    with raises(ValueError):
        AmperometricModel(0.01, 50.0, 0.1, k_analyte=-1.0)
    with raises(ValueError):
        amperometric_current(AmperometricModel(0.01, 50.0, 0.1, k_analyte=1.0), -1.0)
    with raises(ValueError):
        two_line_endpoint([1, 2, 3], [1, 2, 3], 2.0)
    with raises(ValueError):
        two_line_endpoint([1, 2, 3, 7, 8, 9], [1, 2, 3, 7, 8, 9], 5.0)  # one line: parallel


def test_deadstop_current_zero_before_equivalence() -> None:
    y = deadstop_iodine_curve(0.1, 20.0, 0.05, [5.0, 10.0, 19.0, 30.0])
    assert y[0] == 0.0 and y[1] == 0.0 and y[2] == 0.0 and y[3] > 0.0
    assert biamperometric_current(0.0, 1.0, 10.0) == 0.0
    with raises(ValueError):
        biamperometric_current(-1.0, 1.0, 10.0)


# ----------------------------------------------------------- thermometry
def test_thermometric_neutralization() -> None:
    m = ThermometricModel(0.1, 50.0, 0.1, -55.8)
    assert close(thermometric_dt(m, 50.0), 5.0 * 55.8 / (4.18 * 100.0), 1e-12)
    v, t = thermometric_curve(m, n=40)
    assert int(t.argmax()) == int(abs(v - 50.0).argmin()) or t.max() - t[-1] < 0.05
    cold = ThermometricModel(0.1, 50.0, 0.1, -55.8, titrant_dt_k=-3.0)
    v, t = thermometric_curve(cold, n=60, dilution_corrected=True)
    assert abs(two_line_endpoint(v, t, 50.0) - 50.0) < 1e-6
    with raises(ValueError):
        thermometric_dt(m, -1.0)
    with raises(ValueError):
        ThermometricModel(0.1, 50.0, 0.1, -55.8, heat_capacity_j_ml_k=0.0)


# ---------------------------------------------------------------- Karl Fischer
def test_karl_fischer() -> None:
    assert close(KF := kf_titer_mg_per_ml(100.0, 4.0, "sodium_tartrate_dihydrate"), 3.915, 1e-3)
    assert close(kf_titer_mg_per_ml(50.0, 10.0, "water", v_blank_ml=0.0), 5.0)
    assert close(kf_water_percent(3.0, 5.0, 1000.0), 1.5)
    assert kf_water_percent(3.0, 5.0, 1000.0, 10.0, 3.0) < 1.5            # drift lowers the result
    assert abs(coulometric_kf_water_ug(10.711) - 1000.0) < 0.5            # 10.71 C per mg of water
    assert KF > 0
    with raises(ValueError):
        kf_titer_mg_per_ml(100.0, 4.0, "sand")
    with raises(ValueError):
        kf_water_percent(1.0, 5.0, 0.0)


# ------------------------------------------------------------- coulometry
def test_coulometry_faraday() -> None:
    assert close(coulometric_amount_mmol(10.0, 965.0, 1.0), 0.1, 1e-3)
    n = 0.37
    t = coulometric_time_s(n, 5.0, 2.0, 0.998)
    assert close(coulometric_amount_mmol(5.0, t, 2.0, 0.998), n, 1e-12)
    assert close(coulometric_efficiency_bias_percent(0.995), 0.5, 1e-9)
    with raises(ValueError):
        coulometric_amount_mmol(1.0, 1.0, 1.0, 1.2)
    with raises(ValueError):
        coulometric_time_s(1.0, 0.0, 1.0)


# ------------------------------------------------- standardization
def test_standardization_roundtrip() -> None:
    for key, (mw, per_t, _) in PRIMARY_STANDARDS.items():
        v = 25.0
        c_true = 0.1
        mass = c_true * v / 1000.0 * per_t * mw
        assert close(standardize_titrant(key, mass, v), c_true, 1e-9), key
    assert standardize_titrant("khp_naoh", 0.5106, 25.0, purity=0.999) < standardize_titrant("khp_naoh", 0.5106, 25.0)
    with raises(ValueError):
        standardize_titrant("nope", 1.0, 1.0)
    with raises(ValueError):
        standardize_titrant("khp_naoh", 1.0, 1.0, purity=1.5)


# ------------------------------------------------------------------- GUM
def test_gum_matches_closed_form_product_rule() -> None:
    q = [Quantity("c", 0.1, 0.0002), Quantity("vt", 25.0, 0.03), Quantity("va", 25.0, 0.03)]
    r = gum_budget(lambda c, vt, va: c * vt / va, q)
    rel = math.sqrt((0.0002 / 0.1) ** 2 + (0.03 / 25) ** 2 * 2)
    assert close(r.u_c, 0.1 * rel, 1e-6)
    assert abs(sum(x[3] for x in r.contributions) - 100.0) < 1e-9
    assert r.k == 2.0 and close(r.u_expanded, 2 * r.u_c)


def test_gum_welch_satterthwaite_and_k() -> None:
    q = [Quantity("a", 1.0, 0.1, 4), Quantity("b", 1.0, 0.1, 4)]
    r = gum_budget(lambda a, b: a + b, q)
    assert close(r.dof_eff, 8.0, 1e-6) and r.k == 2.37
    assert coverage_factor(4.9) == 2.87 and coverage_factor(math.inf) == 2.0 and coverage_factor(0.5) == 13.97
    with raises(ValueError):
        gum_budget(lambda a: a, [])
    with raises(ValueError):
        Quantity("x", 1.0, -1.0)


def test_gum_helpers_and_titration_budget() -> None:
    assert close(u_rectangular(0.05), 0.05 / math.sqrt(3))
    assert close(u_triangular(0.1), 0.1 / math.sqrt(6))
    assert close(u_expanded_to_standard(0.02, 2.0), 0.01)
    base = titration_gum(0.1, 0.0002, 25.0, 0.03, 25.0, 0.03)
    more = titration_gum(0.1, 0.0002, 25.0, 0.03, 25.0, 0.03, u_endpoint_ml=0.05)
    assert more.u_c > base.u_c
    assert [c[0] for c in more.contributions] == ["C_T", "V_T", "V_A", "end point"]


def test_markdown_report() -> None:
    s = markdown_report("T", [("A", [("x", "1"), ("y", "2")])])
    assert s.startswith("# T") and "| y | 2 |" in s
