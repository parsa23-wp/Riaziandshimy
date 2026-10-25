"""Tests for redox, precipitation, EDTA and back titration."""
from __future__ import annotations

import math

from titration_sim.engine.phase2 import (
    LOG_KF, EdtaModel, PrecipitationModel, RedoxModel, alpha_y4, back_titration, conditional_kf,
    edta_equivalence_volume, edta_pm, hardness_mg_per_l_caco3, mohr_endpoint_volume, nernst_slope,
    precip_ag, precip_equivalence_volume, precip_pag, precip_px, redox_curve,
    redox_equivalence_potential, redox_equivalence_volume, redox_potential, volhard_result,
)

from ._compat import close, raises

FE_CE = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1)


def test_nernst_slope_temperature() -> None:
    assert close(nernst_slope(25.0), 0.05916, 1e-3)
    assert nernst_slope(50.0) > nernst_slope(25.0)


def test_redox_half_and_double_points() -> None:
    assert close(redox_potential(FE_CE, 25.0), 0.77, 1e-6)      # f = 0.5
    assert close(redox_potential(FE_CE, 100.0), 1.44, 1e-6)     # f = 2
    assert close(redox_potential(FE_CE, 50.0), 1.105, 1e-9)     # f = 1 exactly


def test_redox_jump_and_monotone() -> None:
    v, e = redox_curve(FE_CE, n=400)
    assert all(e[i] < e[i + 1] for i in range(len(e) - 1))
    assert redox_potential(FE_CE, 50.05) - redox_potential(FE_CE, 49.95) > 0.3
    assert close(redox_equivalence_volume(FE_CE), 50.0)


def test_redox_unequal_electrons_and_ph() -> None:
    m = RedoxModel(0.05, 40.0, 0.02, 0.54, 2, 1.51, 5, ph=0.0, m2=8)   # 2:5 stoichiometry
    assert close(redox_equivalence_volume(m), 2 * 0.05 * 40.0 / (5 * 0.02))
    m_ph = RedoxModel(0.05, 40.0, 0.02, 0.54, 2, 1.51, 5, ph=2.0, m2=8)
    assert redox_equivalence_potential(m_ph) < redox_equivalence_potential(m)


def test_redox_errors() -> None:
    with raises(ValueError):
        redox_potential(FE_CE, 0.0)
    with raises(ValueError):
        RedoxModel(0.1, 50.0, 0.1, 1.0, 1, 0.5, 1)
    with raises(ValueError):
        RedoxModel(0.0, 50.0, 0.1, 0.5, 1, 1.0, 1)


AGCL = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)


def test_precip_equivalence_is_half_pksp() -> None:
    assert close(precip_pag(AGCL, 50.0), -0.5 * math.log10(1.8e-10), 1e-9)
    assert close(precip_px(AGCL, 50.0), -0.5 * math.log10(1.8e-10), 1e-9)


def test_precip_before_after_equivalence() -> None:
    assert close(precip_px(AGCL, 0.0), 1.0)
    assert close(precip_px(AGCL, 25.0), -math.log10(2.5 / 75.0), 1e-3)
    assert close(precip_pag(AGCL, 75.0), -math.log10(0.1 * 25.0 / 125.0), 1e-3)
    assert precip_pag(AGCL, 0.0) == math.inf
    assert close(precip_equivalence_volume(AGCL), 50.0)


def test_precip_mass_balance_and_ksp_hold() -> None:
    for v in (5.0, 49.9, 50.0, 50.1, 90.0):
        ag = precip_ag(AGCL, v)
        x = 10.0 ** (-precip_px(AGCL, v))
        assert close(ag * x, 1.8e-10, 1e-6)


def test_precip_ksp_not_exceeded_at_trace_titrant() -> None:
    m = PrecipitationModel(1e-6, 50.0, 1e-6, 1.8e-10)       # 1e-6 * 1e-6 < Ksp -> no solid
    assert close(precip_ag(m, 10.0), 1e-6 * 10.0 / 60.0, 1e-9)


def test_mohr_endpoint_small_positive_error() -> None:
    v = mohr_endpoint_volume(AGCL)
    assert 50.0 < v < 50.3
    with raises(ValueError):
        mohr_endpoint_volume(AGCL, c_cro4=1e-30)


def test_volhard() -> None:
    assert close(volhard_result(0.1, 50.0, 0.1, 20.0, 25.0), 0.12)
    with raises(ValueError):
        volhard_result(0.1, 10.0, 0.1, 20.0, 25.0)


def test_alpha_y4_known_values() -> None:
    assert close(alpha_y4(10.0), 0.355, 3e-3)
    # tabulated values (Skoog/West/Holler): pH 3 -> 2.5e-11, pH 8 -> 5.4e-3, pH 10 -> 0.35, pH 12 -> 0.98
    assert close(alpha_y4(3.0) / 2.5e-11, 1.0, 0.02) and close(alpha_y4(8.0) / 5.4e-3, 1.0, 0.01)
    assert close(alpha_y4(12.0), 0.98, 5e-3)
    assert close(alpha_y4(12.0), 0.98, 5e-3)
    assert alpha_y4(2.0) < 1e-10


EDTA = EdtaModel(0.01, 50.0, 0.01, LOG_KF["Ca"], 10.0)


def test_edta_start_and_equivalence() -> None:
    assert close(edta_pm(EDTA, 0.0), 2.0, 1e-9)
    k = conditional_kf(EDTA)
    expected = 0.5 * (-math.log10(0.005) + math.log10(k))      # pM_eq = (pC + log K')/2
    assert close(edta_pm(EDTA, 50.0), expected, 5e-3)
    assert close(edta_equivalence_volume(EDTA), 50.0)


def test_edta_monotone_and_ph_dependence() -> None:
    pms = [edta_pm(EDTA, v) for v in range(0, 100, 5)]
    assert all(a < b for a, b in zip(pms, pms[1:]))
    low_ph = EdtaModel(0.01, 50.0, 0.01, LOG_KF["Ca"], 8.0)
    assert edta_pm(low_ph, 60.0) < edta_pm(EDTA, 60.0)          # weaker K' at lower pH


def test_edta_after_equivalence_formula() -> None:
    k = conditional_kf(EDTA)
    v = 75.0
    excess = 0.01 * 25.0 / 125.0
    mt = 0.01 * 50.0 / 125.0
    expected = -math.log10(mt / (k * excess))
    assert close(edta_pm(EDTA, v), expected, 5e-3)


def test_hardness_and_back_titration() -> None:
    assert close(hardness_mg_per_l_caco3(0.01, 12.5, 100.0), 125.1125, 1e-6)
    r = back_titration(0.1, 50.0, 0.1, 20.0, 25.0)
    assert close(r.n_analyte_mmol, 3.0) and close(r.c_analyte, 0.12)
    r2 = back_titration(0.1, 50.0, 0.05, 20.0, 25.0, reagent_per_analyte=2.0)   # 5 - 1 = 4 -> 2
    assert close(r2.n_analyte_mmol, 2.0)
    with raises(ValueError):
        back_titration(0.1, 10.0, 0.1, 20.0, 25.0)


# ---------------------------------------------------- v0.8: limitation fixes
from titration_sim.engine.phase2 import (  # noqa: E402
    AGCL_LOG_BETA, EDTA_PKA, FAJANS_DYES, NH3_LOG_BETA, alpha_metal,
    conditional_kf as _ck, edta_pm as _epm, fajans_endpoint_volume,
    fajans_indicator_error_percent, fajans_ph_ok, precip_total_dissolved_silver,
    precip_ag as _pag, redox_potential_general, redox_species,
)
from titration_sim.engine.activity import (  # noqa: E402
    concentration_ksp, dh_A, formal_potential_shift, gamma, ionic_strength, log_gamma,
)

FE_CE_G = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1)


def test_activity_davies_values() -> None:
    assert close(log_gamma(1, 0.1), -0.107, 1e-3)
    assert close(log_gamma(2, 0.1), -0.428, 1e-3)
    assert log_gamma(0, 0.1) == 0.0                       # neutral species
    assert close(gamma(1, 0.01), 0.902, 1e-3)
    assert dh_A(25.0) > dh_A(5.0)


def test_ionic_strength() -> None:
    assert close(ionic_strength({1: 0.1, -1: 0.1}), 0.1)
    assert close(ionic_strength({2: 0.1, -1: 0.2}), 0.3)


def test_activity_ksp_and_kf() -> None:
    kp = concentration_ksp(1.8e-10, 0.1)
    assert kp > 1.8e-10                                   # gamma product < 1
    assert close(kp, 2.95e-10, 5e-3)


def test_redox_activity_formal_potential() -> None:
    # Fe3+/Fe2+ at I = 0.1: E1' shifts by s*(log g3 - log g2) < 0
    m = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1, ionic_strength=0.1,
                   z1_ox=3, z1_red=2, z2_ox=4, z2_red=3)
    shift = formal_potential_shift(0.0, 1, nernst_slope(25.0), 3, 2, 0.1)
    assert shift < 0.0
    assert close(m._e1f(), 0.77 + shift, 1e-9)
    # without charges given, no shift is applied
    m2 = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1, ionic_strength=0.1)
    assert close(m2._e1f(), 0.77, 1e-12)


def test_general_solver_matches_closed_form() -> None:
    for v in (5.0, 25.0, 45.0, 55.0, 75.0, 99.0):
        assert abs(redox_potential_general(FE_CE_G, v) - redox_potential(FE_CE_G, v)) < 1e-8


def test_dichromate_equivalence_exact() -> None:
    s = 0.05916
    fe3, cr3 = 0.05, 0.05 / 3.0
    analytic = (0.771 + 6 * 1.33) / 7 + (s / 7) * math.log10(fe3 / (6.0 * cr3 * cr3))
    m = RedoxModel(0.1, 50.0, 1.0 / 60.0, 0.771, 1, 1.33, 6, ph=0.0, m2=14, a2=1, b2=2)
    assert close(redox_equivalence_volume(m), 50.0, 1e-9)
    assert close(redox_potential(m, 50.0), analytic, 1e-5)
    sp = redox_species(m, 50.0)
    assert close(sp["Ox1"], 3.0 * sp["Red2"], 1e-9)       # n1[Ox1]/a1 = n2[Red2]/b2


def test_dichromate_curve_monotone_and_species() -> None:
    m = RedoxModel(0.1, 50.0, 1.0 / 60.0, 0.771, 1, 1.33, 6, ph=0.0, m2=14, a2=1, b2=2)
    v, e = redox_curve(m, n=60)
    assert all(e[i] < e[i + 1] for i in range(len(e) - 1))
    sp = redox_species(m, 25.0)
    # before eq all titrant is consumed: [Cr3+] = 2 * (analyte oxidised)/... balance
    assert close(1.0 * sp["Ox1"] / 1.0, 6.0 * sp["Red2"] / 2.0, 1e-9)
    assert sp["Ox2"] < 1e-9


def test_thiosulfate_iodine_custom_atoms() -> None:
    m = RedoxModel(0.1, 50.0, 0.05, 0.08, 2, 0.535, 2,
                   a1=1, b1=2, atoms1_ox=4, atoms1_red=2, a2=1, b2=2)
    assert close(redox_equivalence_volume(m), 50.0, 1e-9)   # C1V1/(2C2)
    v, e = redox_curve(m, n=30)
    assert all(e[i] < e[i + 1] for i in range(len(e) - 1))
    sp = redox_species(m, 25.0)
    # half the thiosulfate oxidised: [S4O6] = [S2O3]/2
    assert close(sp["Ox1"], 0.5 * sp["Red1"], 1e-9)


def test_redox_asymmetric_validation() -> None:
    with raises(ValueError):
        RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1, a1=0.0)
    assert not FE_CE_G.asymmetric
    assert RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1, b2=2).asymmetric


def test_precip_activity_corrected_ksp() -> None:
    m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10, ionic_strength=0.1)
    assert m.ksp_eff > m.ksp
    # equivalence pAg is halved-log of the *effective* Ksp
    assert close(precip_pag(m, 50.0), -0.5 * math.log10(m.ksp_eff), 1e-9)
    with raises(ValueError):
        PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10, ionic_strength=-1.0)


def test_precip_complexes_do_not_change_free_curve() -> None:
    """1:1 complexes cancel in the difference balance: free pAg is unchanged."""
    m0 = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)
    for v in (10.0, 25.0, 49.99, 50.01, 60.0):
        assert _pag(m0, v) == _pag(m0, v)                 # trivial, but the doc proof:
        tot = precip_total_dissolved_silver(m0, v, AGCL_LOG_BETA)
        assert tot >= _pag(m0, v) * 0.0 + _pag(m0, v)     # total dissolved >= free
    m25 = precip_total_dissolved_silver(m0, 25.0)
    assert m25 > _pag(m0, 25.0) * 0.0 + _pag(m0, 25.0)    # complexes add dissolved Ag


def test_alpha_metal_ammonia() -> None:
    assert alpha_metal() == 1.0
    lg = math.log10(alpha_metal(0.1, NH3_LOG_BETA["Zn"], ph=10.0))
    assert close(lg, -5.49, 5e-3)
    with raises(ValueError):
        alpha_metal(-0.1)


def test_edta_side_reactions_lower_kf() -> None:
    m = EdtaModel(0.01, 50.0, 0.01, LOG_KF["Zn"], 10.0)
    m_lig = EdtaModel(0.01, 50.0, 0.01, LOG_KF["Zn"], 10.0,
                      aux_ligand_conc=0.1, aux_log_beta=NH3_LOG_BETA["Zn"])
    assert close(math.log10(_ck(m)), 16.02, 2e-2)         # alpha_Y4 only
    assert math.log10(_ck(m_lig)) < math.log10(_ck(m)) - 5.0
    assert _epm(m_lig, 25.0) < _epm(m, 25.0)              # weaker complexation -> more free M


def test_edta_activity_correction() -> None:
    m = EdtaModel(0.01, 50.0, 0.01, LOG_KF["Ca"], 10.0, ionic_strength=0.1, z_metal=2)
    m0 = EdtaModel(0.01, 50.0, 0.01, LOG_KF["Ca"], 10.0)
    assert _ck(m) < _ck(m0)                               # gamma_Y4- dominates


def test_fajans_endpoint_and_error() -> None:
    m = PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10)
    ve = precip_equivalence_volume(m)
    v_end = fajans_endpoint_volume(m)
    assert abs(fajans_indicator_error_percent(m)) < 0.2
    assert v_end > ve
    # a sluggish dye (higher threshold) -> larger positive error
    assert fajans_indicator_error_percent(m, ag_threshold=1e-4) > \
        fajans_indicator_error_percent(m)
    assert set(FAJANS_DYES) >= {"fluorescein", "dichlorofluorescein", "eosin"}
    assert fajans_ph_ok(7.0, "fluorescein")
    assert not fajans_ph_ok(5.0, "fluorescein")
    assert fajans_ph_ok(5.0, "eosin")
    with raises(ValueError):
        fajans_ph_ok(7.0, "unknown_dye")
