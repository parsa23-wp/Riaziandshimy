"""Tests of the Qt-free UI layer (run without PyQt6)."""
from __future__ import annotations

from titration_sim.engine.acid_base import AcidBaseModel, equivalence_volumes
from titration_sim.engine.indicators import load_indicators
from titration_sim.ui import colors
from titration_sim.ui.presenter import (
    InputError,
    ModelInput,
    TitrationSession,
    build_model,
    find_indicator,
    make_session,
    parse_number,
    parse_number_list,
)
from titration_sim.ui.strings import LANGS, all_keys, tr

from ._compat import close, raises


def test_parse_number_persian_and_comma() -> None:
    assert parse_number("۰٫۱۲۵") == 0.125
    assert parse_number("2,5") == 2.5
    assert parse_number(" ٣ ") == 3.0
    assert parse_number("−1.5") == -1.5


def test_parse_number_rejects_garbage() -> None:
    for bad in ("", "abc", "nan", "inf", "1..2"):
        with raises(ValueError):
            parse_number(bad)


def test_parse_number_list_separators() -> None:
    assert parse_number_list("2.15 ; 7.20 ۱۲٫۳۵") == (2.15, 7.2, 12.35)
    assert parse_number_list("") == ()


def test_build_model_weak_acid_and_sorting() -> None:
    m = build_model(ModelInput(kind="weak_acid", pka="7.2 2.15 12.35"))
    assert m.analyte_pka == (2.15, 7.2, 12.35)


def test_build_model_errors_name_the_field() -> None:
    cases = [
        (ModelInput(c_analyte="0"), "c_analyte"),
        (ModelInput(v_analyte="x"), "v_analyte"),
        (ModelInput(kind="weak_acid", pka=""), "pka"),
        (ModelInput(titrant_strength="weak", titrant_pka=""), "titrant_pka"),
        (ModelInput(kind="nonsense"), "kind"),
        (ModelInput(pkw="-3"), "pkw"),
    ]
    for inp, key in cases:
        try:
            build_model(inp)
        except InputError as exc:
            assert exc.field_key == key, (key, exc.field_key)
        else:
            raise AssertionError(f"no error for {key}")


def test_find_indicator() -> None:
    assert find_indicator("") is None
    assert find_indicator("Phenolphthalein") is not None
    with raises(InputError):
        find_indicator("Nope")


def test_burette_arithmetic_and_limit() -> None:
    s = TitrationSession(AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0), burette_capacity_ml=30.0)
    assert s.add(10.0) == 10.0
    assert close(s.burette_remaining_fraction, 20.0 / 30.0)
    assert close(s.add(25.0), 20.0)          # only 20 mL left
    assert s.burette_empty and s.add(1.0) == 0.0
    with raises(ValueError):
        s.add(-0.1)
    with raises(ValueError):
        s.add(float("nan"))


def test_session_matches_engine_and_reset() -> None:
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)
    s = TitrationSession(m)
    s.add(12.5)
    assert abs(s.state.ph - 4.76) < 0.02          # half-neutralisation
    n = len(s.states)
    s.reset()
    assert len(s.states) == 1 and s.v_added_ml == 0.0 and n == 2


def test_indicator_colour_changes_across_transition() -> None:
    pp = find_indicator("Phenolphthalein")
    assert pp is not None
    m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    s = TitrationSession(m, pp)
    assert s.flask_color() == colors.WATER or colors.hex_to_rgb(s.flask_color())[2] > 200
    assert s.indicator_changed() is False
    s.add(25.2)
    assert s.indicator_changed() is True
    r, g, b = colors.hex_to_rgb(s.flask_color())
    assert r > g                                   # pink side


def test_indicator_colour_for_base_analyte_is_inverted() -> None:
    mr = find_indicator("Methyl Red")
    m = AcidBaseModel.strong_base(0.1, 25.0, 0.1, pkw=14.0)
    s = TitrationSession(m, mr)
    assert s.indicator_changed() is False          # pH 13 > endpoint pH, falling titration
    s.add(25.5)
    assert s.indicator_changed() is True


def test_no_indicator_gives_water() -> None:
    s = TitrationSession(AcidBaseModel.strong_acid(0.1, 25.0, 0.1))
    assert s.flask_color() == colors.WATER and s.indicator_changed() is None


def test_phenolphthalein_fades_in_strongly_basic_solution() -> None:
    pp = find_indicator("Phenolphthalein")
    assert colors.indicator_color(pp, 13.5) == colors.named_color("colorless")


def test_mix_endpoints_and_clip() -> None:
    assert colors.mix("#000000", "#ffffff", -1) == "#000000"
    assert colors.mix("#000000", "#ffffff", 2) == "#ffffff"


def test_all_indicator_colours_are_known() -> None:
    for ind in load_indicators():
        colors.named_color(ind.acid_color)
        colors.named_color(ind.base_color)


def test_analysis_needs_points_then_finds_veq() -> None:
    m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    s = TitrationSession(m)
    assert s.analysis().v_eq_found is None
    while s.v_added_ml < 30.0:
        s.add(s.suggest_step_ml())
    a = s.analysis()
    assert a.v_eq_found is not None and a.rel_error_percent is not None
    assert abs(a.v_eq_found - 25.0) < 0.1
    assert abs(a.rel_error_percent) < 0.5
    assert a.visual_ok


def test_weak_weak_warns_about_small_jump() -> None:
    m = build_model(ModelInput(kind="weak_acid", titrant_strength="weak", pka="4.76", titrant_pka="9.25"))
    s = TitrationSession(m)
    assert not s.analysis().visual_ok
    assert "potentiometric" in s.analysis().message


def test_auto_step_is_smaller_near_equivalence() -> None:
    s = TitrationSession(AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0))
    far = s.suggest_step_ml()
    s.add(25.0)
    assert s.suggest_step_ml() < far


def test_derivative_and_theory_curve_shapes() -> None:
    s = TitrationSession(AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0))
    assert s.derivative()[0].size == 0
    for _ in range(10):
        s.add(2.0)
    v, d = s.derivative()
    assert v.size == d.size == 11
    tv, tp = s.theory_curve()
    assert tv.size == tp.size > 100 and tv[0] == 0.0


def test_report_languages_and_content() -> None:
    s = make_session(ModelInput(indicator="Methyl Red", kind="weak_acid", pka="4.76"))
    for lang in LANGS:
        txt = s.report(lang)
        assert "pH" in txt and "Methyl Red" in txt
    eq = equivalence_volumes(s.model)[0]
    assert str(round(eq, 3)) in s.report("en")


def test_strings_complete_and_unknown_language() -> None:
    for k in all_keys():
        assert tr(k, "fa") and tr(k, "en")
    with raises(ValueError):
        tr("title", "de")
    with raises(KeyError):
        tr("no_such_key", "en")


def test_make_session_rejects_bad_burette() -> None:
    with raises(InputError):
        make_session(ModelInput(burette_capacity="0"))


# ------------------------------------------------ v0.8: phase-2 families
from titration_sim.ui.presenter import (  # noqa: E402
    BackSession, FAMILIES, FamilySession, build_back_input, build_model,
    build_phase2_model, family_quantity_label, find_family_indicator, make_session,
)
from titration_sim.ui.strings import all_keys  # noqa: E402


def test_build_model_dispatch() -> None:
    assert type(build_model(ModelInput(family="precip", ksp="1.8e-10"))).__name__ == "PrecipitationModel"
    assert type(build_model(ModelInput(family="redox", e1="0.77", n1="1", e2="1.44", n2="1"))).__name__ == "RedoxModel"
    assert type(build_model(ModelInput(family="edta", log_kf="10.65", edta_ph="10"))).__name__ == "EdtaModel"
    assert type(build_model(ModelInput())).__name__ == "AcidBaseModel"


def test_phase2_model_validation_reports_field() -> None:
    try:
        build_phase2_model(ModelInput(family="redox", e1="0.77", n1="1", e2="0.5", n2="1"))
    except InputError as exc:
        assert exc.field_key == "model"
    else:
        raise AssertionError("e2 <= e1 must fail")
    try:
        build_phase2_model(ModelInput(family="edta", log_kf="10.65", edta_ph="10", aux_metal="Xx"))
    except InputError as exc:
        assert exc.field_key == "aux_metal"
    else:
        raise AssertionError("unknown aux metal must fail")


def test_redox_session_end_to_end() -> None:
    inp = ModelInput(family="redox", c_analyte="0.1", v_analyte="25", c_titrant="0.1",
                     e1="0.77", n1="1", e2="1.44", n2="1", indicator="Ferroin")
    s = make_session(inp)
    assert isinstance(s, FamilySession) and s.v_eq == 25.0
    assert family_quantity_label("redox") == "qty_E"
    assert s.current_y() != s.current_y()          # NaN at V = 0 (no Ox yet)
    s.add(12.5)
    assert abs(s.current_y() - 0.77) < 1e-6        # half titrated: E = E1
    assert s.flask_color() == "#c33c54"            # ferroin red below 1.06 V
    s.add(12.5)
    assert abs(s.current_y() - 1.105) < 1e-6
    s.add(12.5)
    assert s.current_y() > 1.3
    assert s.flask_color() == "#9db8c8"            # pale blue after the transition
    assert s.indicator_changed() is True
    s.reset()
    for _ in range(100):                           # realistic fine auto-run sampling
        s.add(0.5)
    a = s.analysis()
    assert a.v_eq_found is not None and abs(a.v_eq_found - 25.0) < 0.2
    assert a.c_found is not None and abs(a.c_found - 0.1) < 2e-3
    assert abs(a.rel_error_percent) < 2.0
    assert "E_eq" in s.report("en") and "pAg" not in s.report("en")


def test_redox_stoichiometry_concentration() -> None:
    inp = ModelInput(family="redox", c_analyte="0.05", v_analyte="40", c_titrant="0.02",
                     e1="0.54", n1="2", e2="1.51", n2="5")
    s = make_session(inp)
    assert abs(s.v_eq - (2 * 0.05 * 40.0 / (5 * 0.02))) < 1e-9
    for _ in range(200):
        s.add(1.0)
    a = s.analysis()
    assert a.c_found is not None and abs(a.c_found - 0.05) / 0.05 < 0.02


def test_precip_session_and_mohr_indicator() -> None:
    inp = ModelInput(family="precip", c_analyte="0.1", v_analyte="25", c_titrant="0.1",
                     ksp="1.8e-10", indicator="Mohr (K2CrO4)")
    s = make_session(inp)
    assert isinstance(s, FamilySession) and s.v_eq == 25.0
    assert family_quantity_label("precip") == "qty_pAg"
    s.add(10.0)
    assert s.current_y() > 8.0                     # pAg large before eq
    assert s.flask_color() == "#e3c93f"            # yellow CrO4
    for _ in range(50):
        s.add(0.5)
    assert s.current_y() < 4.7                     # pAg dropped below the transition
    assert s.flask_color() == "#a2452e"            # red-brown Ag2CrO4
    a = s.analysis()
    assert a.v_eq_found is not None and abs(a.v_eq_found - 25.0) < 0.3
    assert a.c_found is not None and abs(a.c_found - 0.1) < 3e-3


def test_edta_session_colors() -> None:
    # Zn-EDTA at pH 10 WITHOUT auxiliary ligand: pM_eq ~ 9.2 -> EBT flips cleanly
    inp = ModelInput(family="edta", c_analyte="0.01", v_analyte="50", c_titrant="0.01",
                     log_kf="16.5", edta_ph="10", indicator="Eriochrome Black T")
    s = make_session(inp)
    assert isinstance(s, FamilySession) and s.v_eq == 50.0
    s.add(25.0)
    from titration_sim.engine.phase2 import edta_pm
    assert abs(s.current_y() - edta_pm(s.model, 25.0)) < 1e-9   # delegates to the engine
    assert s.current_y() < 6.5
    assert s.flask_color() == "#8e2a4a"            # wine-red with excess metal
    for _ in range(60):
        s.add(1.0)
    assert s.current_y() > 6.5
    assert s.flask_color() == "#2f5fa8"            # blue after the end point
    assert s.indicator_changed() is True


def test_edta_aux_ligand_chemistry() -> None:
    # NH3 lowers the effective constant -> pM at the equivalence point drops
    inp = ModelInput(family="edta", c_analyte="0.01", v_analyte="50", c_titrant="0.01",
                     log_kf="16.5", edta_ph="10", aux_conc="0.1", aux_metal="Zn")
    s = make_session(inp)
    s.add(50.0)
    p_m_eq_ligand = s.current_y()
    assert 5.0 < p_m_eq_ligand < 7.0               # weakened by alpha_M (textbook ~6.4)


def test_back_session_calculator() -> None:
    inp = ModelInput(family="back", c_reagent="0.1", v_reagent="50",
                     c_back_titrant="0.1", v_back_titrant="20", v_sample="25")
    s = make_session(inp)
    assert isinstance(s, BackSession)
    assert s.add(5.0) == 0.0 and s.burette_empty
    assert s.measured()[0].size == 0
    b = build_back_input(inp)
    assert abs(b.n_analyte_mmol() - 3.0) < 1e-9 and abs(b.c_analyte() - 0.12) < 1e-9
    assert "Back" in s.report("en") or "calculator" in s.report("en")
    try:
        build_back_input(ModelInput(family="back", c_reagent="0.1", v_reagent="10",
                                    c_back_titrant="0.1", v_back_titrant="20", v_sample="25"))
    except InputError:
        pass
    else:
        raise AssertionError("negative consumption must fail")


def test_family_indicator_lookup_and_keys() -> None:
    assert find_family_indicator("redox", "Ferroin").transition == 1.06
    assert find_family_indicator("redox", "") is None
    for fam in FAMILIES:
        assert f"family_{fam}" in all_keys()
        assert family_quantity_label(fam) in all_keys()
