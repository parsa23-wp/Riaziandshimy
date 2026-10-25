# ROADMAP / نقشهٔ راه

Status after v0.8 (the client's "do everything remaining" directive): all phase-2 engines, the limitation fixes, Fajans, phase 3 (instrumental) and BOTH UIs are done. / وضعیت پس از v0.۸: همهٔ موتورهای فاز ۲، رفع محدودیت‌ها، Fajans، فاز ۳ (دستگاهی) و هر دو رابط انجام شده است.

| Phase | Content / محتوا | Status |
|---|---|---|
| 1 | Acid–base: strong/strong, weak acid/strong base, strong acid/weak base, weak/weak (small-jump warning), ≥ 6 indicators (10 included), pH–V curve, 1st/2nd derivative + Gran, error model, uncertainty propagation, mean/SD/RSD/SE/CI | **done** (Part 1) |
| 1 (+) | Polyprotic (any n) and potentiometric refinement of Berto et al. 2024 | done, beyond MVP (see CONTRADICTIONS #2) |
| 1 UI | PyQt6 desktop: burette, flask, curve, controls, main window, fa/en | **done** (Part 3) — Qt widgets verified Qt-free; run `pytest tests/ui` on your machine (MATH/CONTRADICTIONS #16) |
| 2 | Redox (Nernst; spec §6.3–6.5) + asymmetric couples (Cr2O7, I2), pH shift, activity correction | **done** (`engine/phase2.py` + `engine/activity.py`, general electron-balance solver, verified 1.262627 V dichromate) |
| 2 | Precipitation, Ksp (Mohr, Volhard, **Fajans**; spec §7) + activity-corrected Ksp + 1:1-complex proof | **done** (`fajans_endpoint_volume`, `precip_total_dissolved_silver`) |
| 2 | Complexometry EDTA, K'' = Kf·αY·αM·γ, auxiliary ligand (NH3 table), hydrolysis (user beta), hardness (spec §8) | **done** (`alpha_metal`, `NH3_LOG_BETA`) |
| 2 | Back titration workflow (spec §9) | **done** (engine + desktop calculator family + web calculator) |
| 3 | Instrumental: potentiometric EMF (SCE/Ag-AgCl), conductometric (λ° table + T correction), photometric (Beer-Lambert, permanganate curve) | **done** (`engine/phase3.py`) |
| UI | Phase-2 families in the PyQt6 desktop (family selector, generic E/pAg/pM curves, family indicators, back-titration calculator) | **done** (`ui/presenter.py` `FamilySession`, fa/en strings) |
| UI | Phase-2 families in the web `titration-lab.html` (family selector, 6 presets, derivative view, family indicator table, EN toggle of the new block) | **done** (browser-verified: Ce, dichromate, Mohr, EDTA, back) |
| 4 | Analytical quantities from the client's titration text (titer, assay, factor, concordance, LOD/LOQ, buffer capacity, half-equivalence pKa, non-aqueous calculations, CN masking, carbonate Gran) + Berto data-sufficiency check | **done** (`engine/quantities.py`, web tab «محاسبات کمی», tests/test_quantities.py) |
| 5 | Non-aqueous (pKs table, apparent-pKa curves, jump), amperometry (+ dead stop), thermometry, Karl Fischer (volumetric + coulometric), coulometry (Faraday), titrant standardization (6 primary standards), GUM budget (Welch-Satterthwaite) + Markdown report | **done** (`engine/extended.py`, `tests/test_extended.py`, web tab «غیرآبی و روش‌های ابزاری», browser cross-check JS vs Python) |
| infra | Offline use (no external resources in the HTML), CI workflow (`.github/workflows/ci.yml`: pytest+cov, mypy, real-browser test, `node --check`) | **written; CI itself not run** (no GitHub runner here) |
| UI | New v1.0 families in the PyQt6 desktop | **not done** — engine API only |
| future | Virtual workshop (guided exercises), PDF/Word report export, SIT/Pitzer, redox electrode kinetics, non-1:1 analyte salts, photometric EDTA with metallochromic indicators | listed, not coded |

How a new family plugs in / چطور خانوادهٔ جدید اضافه می‌شود:
1. subclass `TitrationModel` (frozen dataclass) with its constants;
2. subclass `BaseEngine` implementing the pure `titrate(model, state, dV)`;
3. the UI already talks to `BaseEngine` only; add a `TitrationSession`-like adapter in `ui/presenter.py` and one controls group.

The web app (`titration-lab.html`, v2.2) runs the acid–base model, the logK fit, the error lab **and — since v0.8 — all phase-2 families** (redox with asymmetric couples, precipitation, EDTA, back-titration calculator), cross-checked against the Python engine (dichromate E_eq 1.262627 V, pAg_eq 4.87, back result 3 mmol). Any new family only needs the three steps above. / وب‌اپ `titration-lab.html` (v2.2) علاوه بر اسید–باز و آزمایشگاه خطا، از v0.۸ همهٔ خانواده‌های فاز ۲ را هم دارد و با موتور پایتون تطبیق داده شده است.
