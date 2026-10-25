"""Tests for engine/quantities.py (brief sections 16, 21-28, 50-96) and the
new error sources (air bubble, parallax) in engine/errors.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from titration_sim.engine.acid_base import AcidBaseModel
from titration_sim.engine.errors import ErrorConfig, simulate_replicate
from titration_sim.engine.quantities import (
    CN_LOG_BETA,
    HCN_PKA,
    assay_percent,
    buffer_capacity,
    carbonate_from_two_endpoints,
    concordant_results,
    edta_masking_selectivity,
    factor_f,
    hclo4_from_khp,
    lod_loq,
    nonaqueous_assay_percent,
    pka_from_half_equivalence,
    resolvable_in_solvent,
    species_coverage_check,
    titer_mg_per_ml,
)


# -- titer / assay / factor ------------------------------------------------

def test_titer_basic_and_ratio() -> None:
    assert titer_mg_per_ml(0.1, 36.46) == pytest.approx(3.646)
    assert titer_mg_per_ml(0.1, 100.09, ratio=0.5) == pytest.approx(5.0045)
    with pytest.raises(ValueError):
        titer_mg_per_ml(-0.1, 36.46)


def test_assay_percent_with_aliquot() -> None:
    # 0.15 g sample, 0.1 M HCl, 25.00 mL, CaCO3 (1:2), full flask
    assert assay_percent(0.15, 0.1, 0.025, 100.09, ratio=0.5) == pytest.approx(83.4083, rel=1e-4)
    # same but only half of the flask titrated -> double
    full = assay_percent(0.15, 0.1, 0.025, 100.09, ratio=0.5)
    aliq = assay_percent(0.15, 0.1, 0.025, 100.09, ratio=0.5, v_flask_l=0.25, v_aliquot_l=0.125)
    assert aliq == pytest.approx(2 * full)
    with pytest.raises(ValueError):
        assay_percent(0.15, 0.1, 0.025, 100.09, v_flask_l=0.1, v_aliquot_l=0.2)


def test_hclo4_standardization_and_factor() -> None:
    assert hclo4_from_khp(0.5106, 0.025) == pytest.approx(0.10003, rel=1e-3)
    assert hclo4_from_khp(0.5106, 0.025, purity_khp=0.995) == pytest.approx(0.09953, rel=1e-3)
    assert factor_f(0.1023, 0.1) == pytest.approx(1.023)
    with pytest.raises(ValueError):
        factor_f(1.0, 0.0)


def test_nonaqueous_assay_matches_core() -> None:
    assert nonaqueous_assay_percent(0.2, 0.02, 0.1, 122.12, ratio=0.5) == pytest.approx(
        assay_percent(0.2, 0.1, 0.02, 122.12, ratio=0.5)
    )


def test_resolvable_in_solvent() -> None:
    ok, gap = resolvable_in_solvent([10.0, 6.0, 2.0])
    assert ok and gap == pytest.approx(4.0)
    ok2, gap2 = resolvable_in_solvent([10.0, 8.5])
    assert not ok2 and gap2 == pytest.approx(1.5)
    with pytest.raises(ValueError):
        resolvable_in_solvent([7.0])


# -- concordance / LOD / statistics ----------------------------------------

def test_concordance_and_float_guard() -> None:
    c = concordant_results([24.82, 24.79, 24.84])
    assert c.ok and c.spread == pytest.approx(0.05, abs=1e-9)
    assert not concordant_results([24.82, 25.30]).ok
    with pytest.raises(ValueError):
        concordant_results([1.0])


def test_lod_loq_styles() -> None:
    ich = lod_loq(0.002, 1.5)
    assert ich.lod == pytest.approx(3.3 * 0.002 / 1.5)
    assert ich.loq == pytest.approx(10 * 0.002 / 1.5)
    classic = lod_loq(0.002, 1.5, style="classic")
    assert classic.lod == pytest.approx(3 * 0.002 / 1.5)
    with pytest.raises(ValueError):
        lod_loq(0.002, 1.5, style="other")
    with pytest.raises(ValueError):
        lod_loq(0.002, -1.0)


# -- buffer capacity / half-equivalence pKa --------------------------------

def test_buffer_capacity_max_at_pka() -> None:
    ka = 10.0 ** -4.76
    at_pka = buffer_capacity(ka, ka, 0.1)
    assert at_pka == pytest.approx(2.302585 * 0.1 * 0.25, rel=1e-3)
    # strictly lower one pH unit away
    away = buffer_capacity(ka / 10.0, ka, 0.1)
    assert away < at_pka


def test_pka_from_half_equivalence() -> None:
    ph = pka_from_half_equivalence([0, 5, 10, 15], [3.0, 4.76, 6.0, 8.0], 0, 10)
    assert ph == pytest.approx(4.76)
    with pytest.raises(ValueError):
        pka_from_half_equivalence([0, 5], [3.0, 4.0], 0, 100)


# -- carbonate by double end point (Berto 2.6(ii)) --------------------------

def test_carbonate_two_endpoints() -> None:
    c_co3, c_oh = carbonate_from_two_endpoints(20.0, 30.0, 0.1, 50.0)
    assert c_co3 == pytest.approx(0.02)
    assert c_oh == pytest.approx(0.02)
    with pytest.raises(ValueError):
        carbonate_from_two_endpoints(10.0, 25.0, 0.1, 50.0)  # 2 V1 < V2


# -- Berto data sufficiency --------------------------------------------------

def test_species_coverage_good_and_bad() -> None:
    good = species_coverage_check([4.76 + 0.05 * i for i in range(-10, 11)], [4.76])
    assert good.adequate and good.per_species == (21, 21)
    bad = species_coverage_check([2.0, 7.0, 12.0], [4.76])
    assert not bad.adequate


# -- EDTA masking / selectivity ----------------------------------------------

def test_cn_masks_zn_cd_cu_ni_not_ca_mg() -> None:
    r = edta_masking_selectivity(10.0, 0.01)
    for m in ("Zn2+", "Cd2+", "Cu2+", "Ni2+", "Hg2+"):
        assert not r[m][2], m
    for m in ("Ca2+", "Mg2+", "Pb2+", "Mn2+"):
        assert r[m][2], m


def test_masking_without_ligand_all_titratable() -> None:
    r = edta_masking_selectivity(10.0, 0.0)
    assert all(v[2] for v in r.values())


def test_cn_free_fraction_refinement() -> None:
    # at pH 6 nearly all CN- is HCN -> far less masking than at pH 10
    r6 = edta_masking_selectivity(6.0, 0.01, ligand_pka=HCN_PKA)
    assert r6["Zn2+"][2]  # masking lost at low pH
    # and the free-fraction correction genuinely reduced the ligand strength
    assert r6["Zn2+"][0] > edta_masking_selectivity(10.0, 0.01, ligand_pka=HCN_PKA)["Zn2+"][0] - 1e-9


def test_cn_table_metal_missing_is_unmasked() -> None:
    r = edta_masking_selectivity(10.0, 0.01, metals_log_kf={"Ca2+": 10.65, "Fe3+": 25.10})
    assert r["Fe3+"][2] and r["Ca2+"][2]  # no CN entry for Fe3+ -> alpha_M = 1


# -- air bubble / parallax error sources --------------------------------------

M = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)


def test_bubble_biases_high() -> None:
    base = simulate_replicate(M, None, ErrorConfig(), np.random.default_rng(1))
    bub = simulate_replicate(M, None, ErrorConfig(bubble_ml=0.05), np.random.default_rng(1))
    # c_calc = C_t * V_read / (k V_a) -> +0.05 mL reading means + C_t*0.05/V_a
    assert bub.c_calc == pytest.approx(base.c_calc + M.c_titrant * 0.05 / M.v_analyte_ml, rel=1e-9)
    assert bub.v_read_ml == pytest.approx(base.v_read_ml + 0.05)


def test_parallax_adds_noise_same_seed_deterministic() -> None:
    a = simulate_replicate(M, None, ErrorConfig(sigma_parallax_ml=0.03), np.random.default_rng(7))
    b = simulate_replicate(M, None, ErrorConfig(sigma_parallax_ml=0.03), np.random.default_rng(7))
    assert a.c_calc == b.c_calc  # deterministic generator
    no_p = simulate_replicate(M, None, ErrorConfig(), np.random.default_rng(7))
    assert not math.isclose(a.v_read_ml, no_p.v_read_ml)  # the noise actually acts
