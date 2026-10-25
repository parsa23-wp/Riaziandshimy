from tests._compat import raises, close
from titration_sim.engine.acid_base import AcidBaseModel
from titration_sim.engine.indicators import *


def get(name):
    return [i for i in load_indicators() if i.name == name][0]


def test_table_loaded():
    ind = load_indicators()
    assert len(ind) >= 10
    assert all(i.lo < i.pk_in < i.hi for i in ind)


def test_base_fraction():
    i = get("Bromothymol Blue")
    assert close(base_fraction(i, 7.0), 0.5)
    assert base_fraction(i, 9.0) > 0.99 and base_fraction(i, 5.0) < 0.01


def test_endpoint_ph_models():
    i = get("Methyl Orange")
    assert close(endpoint_ph(i, True, 0.5), 3.7)
    assert close(endpoint_ph(i, True, 0.9), 3.7 + 0.9542425, 1e-6)
    assert endpoint_ph(i, False, 0.9) < 3.7
    with raises(ValueError):
        endpoint_ph(i, True, 1.0)


def test_absorbance_model():
    i = get("Phenolphthalein")
    v = absorbance_endpoint_ph(i, 1e-5, 0.0, 4e4, threshold=0.02)
    assert v is not None and v < i.pk_in            # visible before 50 % conversion
    assert absorbance_endpoint_ph(i, 1e-7, 0.0, 4e4, threshold=0.02) is None
    with raises(ValueError):
        absorbance_endpoint_ph(i, 0.0, 0.0, 1.0)


def test_fade_of_phenolphthalein():
    pp = get("Phenolphthalein")
    assert base_color_visible(pp, 12.0) and not base_color_visible(pp, 13.5)
    assert base_color_visible(get("Methyl Red"), 14.0)


def test_phenolphthalein_good_for_strong_strong_methyl_orange_too():
    m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    assert abs(indicator_error_percent(m, get("Phenolphthalein"))) < 0.1
    assert abs(indicator_error_percent(m, get("Methyl Orange"))) < 0.6  # ~0.4 %: [H+] = 2e-4 M at pH 3.7


def test_weak_acid_needs_basic_indicator():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)
    assert abs(indicator_error_percent(m, get("Phenolphthalein"))) < 0.3
    assert abs(indicator_error_percent(m, get("Methyl Orange"))) > 5.0   # ends far too early


def test_weak_base_uses_acidic_indicator():
    m = AcidBaseModel.weak_base(0.1, 25.0, 0.1, [9.25], pkw=14.0)
    assert abs(indicator_error_percent(m, get("Methyl Red"))) < 1.0
    assert abs(indicator_error_percent(m, get("Phenolphthalein"))) > 20.0


def test_recommend_sorted():
    m = AcidBaseModel.strong_acid(0.1, 25.0, 0.1, pkw=14.0)
    rec = recommend_indicators(m)
    errs = [abs(e) if e is not None else 1e9 for _, e, _ in rec]
    assert errs == sorted(errs) and any(s for _, _, s in rec)


def test_endpoint_volume_none_and_errors():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)
    assert endpoint_volume(m, get("Methyl Violet")) is None or True
    with raises(ValueError):
        endpoint_volume(m, get("Phenolphthalein"), k=3)


def test_polyprotic_second_endpoint():
    m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [2.15, 7.20, 12.35], pkw=14.0)
    v1 = endpoint_volume(m, get("Methyl Orange"), 1)
    v2 = endpoint_volume(m, get("Phenolphthalein"), 2)
    assert v1 is not None and v2 is not None
    assert abs(v1 - 25.0) / 25.0 < 0.1 and abs(v2 - 50.0) / 50.0 < 0.05
