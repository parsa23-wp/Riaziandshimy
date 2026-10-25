"""Bilingual (fa/en) UI strings.  Keys are stable identifiers; add a language by adding a column."""
from __future__ import annotations

LANGS = ("fa", "en")

_S: dict[str, tuple[str, str]] = {
    "title": ("شبیه‌ساز تیتراسیون", "Titration simulator"),
    "analyte": ("آنالیت", "Analyte"),
    "titrant": ("تیترانت", "Titrant"),
    "kind_strong_acid": ("اسید قوی", "Strong acid"),
    "kind_weak_acid": ("اسید ضعیف", "Weak acid"),
    "kind_strong_base": ("باز قوی", "Strong base"),
    "kind_weak_base": ("باز ضعیف (pKa اسید مزدوج)", "Weak base (pKa of conjugate acid)"),
    "titrant_strong": ("قوی", "Strong"),
    "titrant_weak": ("ضعیف", "Weak"),
    "c_analyte": ("غلظت آنالیت (mol/L)", "Analyte concentration (mol/L)"),
    "v_analyte": ("حجم آنالیت (mL)", "Analyte volume (mL)"),
    "c_titrant": ("غلظت تیترانت (mol/L)", "Titrant concentration (mol/L)"),
    "pka": ("pKa (چند مقدار با فاصله یا ;)", "pKa (several values separated by space or ;)"),
    "titrant_pka": ("pKa تیترانت ضعیف", "pKa of weak titrant"),
    "temperature": ("دما (°C)", "Temperature (°C)"),
    "pkw": ("pKw (خالی = خودکار از دما)", "pKw (empty = automatic from T)"),
    "indicator": ("شناساگر", "Indicator"),
    "no_indicator": ("بدون شناساگر (pH‌متر)", "No indicator (pH meter)"),
    "apply": ("اعمال و شروع دوباره", "Apply and restart"),
    "reset": ("شروع دوباره", "Restart"),
    "drop": ("+۰٫۰۵ mL (یک قطره)", "+0.05 mL (one drop)"),
    "add_01": ("+۰٫۱ mL", "+0.1 mL"),
    "add_1": ("+۱ mL", "+1 mL"),
    "add_5": ("+۵ mL", "+5 mL"),
    "custom_dv": ("مقدار دلخواه (mL)", "Custom amount (mL)"),
    "add_custom": ("افزودن", "Add"),
    "auto_start": ("اجرای خودکار", "Auto-run"),
    "auto_stop": ("توقف", "Stop"),
    "show_theory": ("نمایش منحنی نظری", "Show theoretical curve"),
    "burette": ("بورت", "Burette"),
    "reading": ("قرائت", "Reading"),
    "flask": ("ارلن", "Flask"),
    "curve": ("منحنی تیتراسیون", "Titration curve"),
    "deriv": ("مشتق اول", "First derivative"),
    "results": ("نتایج", "Results"),
    "lang": ("English", "فارسی"),
    "burette_empty": ("بورت خالی شد.", "The burette is empty."),
    "input_error": ("ورودی نامعتبر", "Invalid input"),
    "ph_now": ("pH فعلی", "Current pH"),
    "v_added": ("حجم افزوده‌شده", "Volume added"),
    "v_eq_theory": ("حجم هم‌ارزی نظری", "Theoretical equivalence volume"),
    "v_eq_found": ("حجم هم‌ارزی از منحنی", "Equivalence volume from curve"),
    "c_found": ("غلظت محاسبه‌شده", "Computed concentration"),
    "rel_err": ("خطای نسبی", "Relative error"),
    "ph_jump": ("جهش pH (±۰٫۱٪)", "pH jump (±0.1 %)"),
    "ind_changed": ("شناساگر تغییر رنگ داد.", "The indicator has changed colour."),
    "ind_not_yet": ("شناساگر هنوز تغییر رنگ نداده.", "The indicator has not changed colour yet."),
    "ind_error": ("خطای شناساگر", "Indicator error"),
    "need_points": ("برای تحلیل منحنی، پس از هم‌ارزی هم داده بگیرید (حداقل ۷ نقطه).",
                    "Collect points past the equivalence point too (at least 7 points)."),
    # ---- v0.8: phase-2 families ------------------------------------------
    "family": ("خانواده تیتراسیون", "Titration family"),
    "family_acid_base": ("اسید و باز", "Acid-base"),
    "family_redox": ("ردوکس", "Redox"),
    "family_precip": ("رسوبی (نقره‌سنجی)", "Precipitation (argentometry)"),
    "family_edta": ("کمپلکس‌سنجی EDTA", "Complexometry (EDTA)"),
    "family_back": ("تیتراسیون برگشتی (حسابگر)", "Back titration (calculator)"),
    "e1": ("پتانسیل استاندارد جفت آنالیت E°₁ (V)", "Analyte couple standard potential E°1 (V)"),
    "n1": ("الکترون‌های جفت آنالیت n₁", "Analyte couple electrons n1"),
    "e2": ("پتانسیل استاندارد جفت تیترانت E°₂ (V)", "Titrant couple standard potential E°2 (V)"),
    "n2": ("الکترون‌های جفت تیترانت n₂", "Titrant couple electrons n2"),
    "redox_ph": ("pH (برای جابه‌جایی فرمال با m₁/m₂؛ خالی = بدون اثر pH)", "pH (formal shift via m1/m2; empty = pH-independent)"),
    "m1": ("پروتون‌های نیم‌واکنش آنالیت m₁", "Protons of analyte half-reaction m1"),
    "m2": ("پروتون‌های نیم‌واکنش تیترانت m₂", "Protons of titrant half-reaction m2"),
    "ionic_strength": ("قدرت یونی I (mol/L؛ خالی یا ۰ = تصحیح فعالیت نشود)", "Ionic strength I (mol/L; empty or 0 = no activity correction)"),
    "ksp": ("Ksp نمک AgX (هم‌دما)", "Ksp of the AgX salt (thermodynamic)"),
    "log_kf": ("log Kf کمپلکس MY", "log Kf of the MY complex"),
    "edta_ph": ("pH بافر", "Buffer pH"),
    "aux_conc": ("غلظت لیگاند جانبی آزاد (مثلاً NH₃؛ خالی = هیچ)", "Free auxiliary ligand concentration (e.g. NH3; empty = none)"),
    "aux_metal": ("متال با جدول β آمینی (Zn/Cu/Ni/Ag/Cd/Co2؛ خالی = بدون جدول)", "Metal for the built-in NH3 beta table (Zn/Cu/Ni/Ag/Cd/Co2; empty = none)"),
    "c_reagent": ("غلظت معرف اضافی (mol/L)", "Excess-reagent concentration (mol/L)"),
    "v_reagent": ("حجم معرف اضافی (mL)", "Excess-reagent volume (mL)"),
    "c_back_titrant": ("غلظت تیترانت برگشت (mol/L)", "Back-titrant concentration (mol/L)"),
    "v_back_titrant": ("حجم تیترانت برگشت مصرفی (mL)", "Back-titrant volume consumed (mL)"),
    "v_sample": ("حجم نمونه (mL)", "Sample volume (mL)"),
    "ratio_ra": ("مول معرف به ازای هر مول آنالیت", "Mol reagent per mol analyte"),
    "ratio_tr": ("مول تیترانت به ازای هر مول معرف", "Mol titrant per mol reagent"),
    "qty_ph": ("pH", "pH"),
    "qty_E": ("پتانسیل E (V)", "Potential E (V)"),
    "qty_pAg": ("pAg", "pAg"),
    "qty_pM": ("pM", "pM"),
    "jump_redox": ("جهش پتانسیل (±۰٫۱٪ هم‌ارزی)", "Potential jump (±0.1 % of V_eq)"),
    "jump_pag_pm": ("جهش pM/pAg (±۰٫۱٪ هم‌ارزی)", "pM/pAg jump (±0.1 % of V_eq)"),
    "indicator_redox": ("شناساگر ردوکس", "Redox indicator"),
    "indicator_precip": ("شناساگر رسوبی", "Precipitation indicator"),
    "indicator_edta": ("شناساگر کمپلکس‌سنجی", "Complexometric indicator"),
    "no_indicator_generic": ("بدون شناساگر (دستگاهی)", "No indicator (instrumental)"),
    "back_result": ("نتیجه تیتراسیون برگشتی", "Back-titration result"),
    "back_n_analyte": ("مول آنالیت (mmol)", "Amount of analyte (mmol)"),
    "back_c_analyte": ("غلظت آنالیت (mol/L)", "Analyte concentration (mol/L)"),
    "back_consumed": ("معرف مصرف‌شده (mmol)", "Reagent consumed (mmol)"),
    "back_no_curve": ("این خانواده حسابگر است؛ منحنی و بورت ندارد. مقادیر را وارد و «اعمال» را بزنید.",
                      "This family is a calculator: no burette or curve. Fill the fields and press Apply."),
    "redox_kinetics_note": ("مدل نرنست (تعادلی) است؛ سینتیک الکترودی در نظر گرفته نشده.",
                            "Nernstian (equilibrium) model; electrode kinetics are not included."),
}


def tr(key: str, lang: str = "fa") -> str:
    """Translate ``key``.

    Raises
    ------
    KeyError
        Unknown key.
    ValueError
        Unknown language.

    Example
    -------
    >>> tr("reset", "en")
    'Restart'
    """
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}")
    return _S[key][LANGS.index(lang)]


def all_keys() -> list[str]:
    """All string keys (used by the tests to check completeness).

    Example
    -------
    >>> "title" in all_keys()
    True
    """
    return list(_S)
