"""Tests for the instrumental layers (phase 3) and the family dispatcher."""
from __future__ import annotations

import math

import numpy as np

from titration_sim.engine.acid_base import AcidBaseModel
from titration_sim.engine.phase2 import RedoxModel
from titration_sim.engine.phase3 import (
    LAMBDA_25, REF_ELECTRODES, absorbance, conductometric_curve_acid_base,
    conductometric_species_acid_base, conductivity_us_cm, emf_mv,
    permanganate_curve, potentiometric_curve_redox, quantity_of,
)

from ._compat import close, raises

FE_CE = RedoxModel(0.1, 50.0, 0.1, 0.77, 1, 1.44, 1)


# ------------------------------------------------------------ potentiometry
def test_emf_reference_electrodes() -> None:
    assert close(emf_mv(1.105, "SCE"), 861.0, 1e-9)
    assert close(emf_mv(0.443, "AgAgCl_satKCl"), 244.0, 1e-9)
    assert close(emf_mv(0.5, "NHE"), 500.0, 1e-9)
    assert emf_mv(1.0, "SCE", junction_mv=3.0) == emf_mv(1.0, "SCE") + 3.0
    with raises(ValueError):
        emf_mv(1.0, "unknown_ref")


def test_potentiometric_curve_shape() -> None:
    v, emf = potentiometric_curve_redox(FE_CE, n=60)
    assert all(emf[i] < emf[i + 1] for i in range(len(emf) - 1))
    # half-way point: E = E1 -> EMF = E1 - E_ref
    i_half = int(np.argmin(np.abs(v - 25.0)))
    assert close(emf[i_half], (0.77 - REF_ELECTRODES["SCE"]) * 1000.0, 2.0)
    with raises(ValueError):
        potentiometric_curve_redox(FE_CE, n=10, reference="bad")


def test_potentiometric_noise_is_deterministic() -> None:
    v1, e1 = potentiometric_curve_redox(FE_CE, n=40, noise_mv=0.5, seed=11)
    v2, e2 = potentiometric_curve_redox(FE_CE, n=40, noise_mv=0.5, seed=11)
    assert np.array_equal(e1, e2)
    assert np.max(np.abs(e1 - potentiometric_curve_redox(FE_CE, n=40)[1])) > 0.0


# ------------------------------------------------------------ conductometry
def test_conductivity_values_and_temperature() -> None:
    assert close(conductivity_us_cm({"H+": 0.1, "Cl-": 0.1}), 42600.0, 1e-6)
    assert close(conductivity_us_cm({"Na+": 0.1, "Cl-": 0.1}), 12645.0, 1e-6)
    assert close(conductivity_us_cm({"H+": 0.1, "Cl-": 0.1}, 35.0), 42600.0 * 1.19, 1e-6)
    with raises(ValueError):
        conductivity_us_cm({"Xx+": 0.1})
    with raises(ValueError):
        conductivity_us_cm({"H+": -0.1})
    with raises(ValueError):
        conductivity_us_cm({"H+": 0.1}, 99.0)


def test_conductometric_species_strong_acid() -> None:
    m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1)
    conc, lam = conductometric_species_acid_base(m, 0.0)
    assert close(conc["H+"], 0.1, 1e-9) and close(conc["Cl-"], 0.1, 1e-9)
    assert close(conc["OH-"], 1e-13, 1e-12)
    assert lam["H+"] == LAMBDA_25["H+"]
    # at the equivalence point: only NaCl + water
    conc2, _ = conductometric_species_acid_base(m, 25.0)
    assert conc2["H+"] < 1e-6 and close(conc2["Na+"], 0.05, 1e-9)
    assert close(conc2["Cl-"], 0.05, 1e-9)


def test_conductometric_species_weak_acid() -> None:
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76])
    conc, _ = conductometric_species_acid_base(m, 20.0)
    # the anion of the weak acid must be present with the Na+ spectator
    assert conc["analyte(A)"] > 0.0 and conc["Na+"] > 0.0
    # neutral HAc is not counted
    assert all("H1A" not in k for k in conc)


def test_conductometric_curve_v_shape() -> None:
    m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1)
    v, k = conductometric_curve_acid_base(m, n=60)
    assert k[0] > k[30] < k[-1]                    # classic V
    assert close(k[0], 42600.0, 1e-6)              # pure 0.1 M HCl
    # temperature correction propagates
    _, k35 = conductometric_curve_acid_base(m, n=20, temperature_c=35.0)
    assert k35[0] > k[0] * 0 + k[0]


def test_conductometric_weak_base_rises() -> None:
    m = AcidBaseModel.weak_base(0.1, 25.0, 0.1, [9.25])
    v, k = conductometric_curve_acid_base(m, n=40)
    assert k[-1] > k[0]                            # NH4Cl conducts better than NH3


# -------------------------------------------------------------- photometry
def test_absorbance_beer_lambert() -> None:
    assert close(absorbance({"MnO4-": 1e-4}, {"MnO4-": 2400.0}), 0.24, 1e-9)
    assert absorbance({"MnO4-": 1e-4}, {}) == 0.0              # non-absorbing ignored
    assert close(absorbance({"A": 1e-3, "B": 2e-3}, {"A": 100.0, "B": 50.0}, 0.5), 0.1, 1e-9)
    with raises(ValueError):
        absorbance({"A": -1e-3}, {"A": 100.0})
    with raises(ValueError):
        absorbance({"A": 1e-3}, {"A": 100.0}, path_cm=0.0)


def test_permanganate_curve_endpoint() -> None:
    v, a = permanganate_curve(FE_CE, n=80)
    veq = 50.0
    i_before = int(np.argmin(np.abs(v - 0.8 * veq)))
    i_after = int(np.argmin(np.abs(v - 1.5 * veq)))
    assert a[i_before] < 1e-7                      # leak-level absorbance before eq
    # after eq: A = eps * (Ct V - n-ratio excess)/... — linear in V
    excess = 0.1 * (v[i_after] - veq) / (50.0 + v[i_after])
    assert close(a[i_after], 2400.0 * excess, 1e-4)


# ------------------------------------------------------------- dispatcher
def test_quantity_dispatch() -> None:
    assert close(quantity_of(AcidBaseModel.strong_acid(0.1, 25.0, 0.1), 0.0), 1.0, 1e-6)
    assert close(quantity_of(FE_CE, 25.0), 0.77, 1e-6)
    assert math.isnan(quantity_of(FE_CE, 0.0))
    from titration_sim.engine.phase2 import EdtaModel, PrecipitationModel, LOG_KF
    assert close(quantity_of(PrecipitationModel(0.1, 50.0, 0.1, 1.8e-10), 50.0),
                 -0.5 * math.log10(1.8e-10), 1e-9)
    assert quantity_of(EdtaModel(0.01, 50.0, 0.01, LOG_KF["Ca"], 10.0), 0.0) > 1.9
    with raises(TypeError):
        quantity_of(object(), 1.0)  # type: ignore[arg-type]
