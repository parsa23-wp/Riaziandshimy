# titration-sim — v0.9 (all phases + analytical quantities): engine + desktop UI · موتور و رابط

**EN.** Stateless titration engine for acid–base systems (strong/weak, mono/polyprotic, strong or weak titrant,
temperature), indicator model, derivative/Gran/statistics tools, laboratory-error simulation, and an implementation of
the potentiometric refinement of Berto et al. (Anal. Chim. Acta 1303 (2024) 342476, CC BY) with its benchmark.
**FA.** موتور بدون‌حالت برای تیتراسیون اسید و باز (قوی/ضعیف، تک و چندپروتونه، تیترانت قوی یا ضعیف، دما)، مدل شناساگر،
ابزار مشتق/گران/آمار، شبیه‌سازی خطاهای آزمایشگاهی، و پیاده‌سازی برازش پتانسیومتری مقالهٔ Berto و همکاران همراه با دیتاست آزمون.

## Use / استفاده
```python
from titration_sim.engine.acid_base import AcidBaseModel, AcidBaseEngine, generate_curve, check_visual_endpoint
m = AcidBaseModel.weak_acid(0.1, 25.0, 0.1, [4.76], pkw=14.0)       # acetic acid + NaOH
eng = AcidBaseEngine(); s = eng.titrate(m, eng.initial_state(m), 12.5)   # pure step: state_after = titrate(model, state_before, dV)
v, ph = generate_curve(m, 30.0)                                      # adaptive (non-uniform) curve
print(s.ph, check_visual_endpoint(m).message)
```
Paper refinement: `from titration_sim.engine import potentiometry as P` → `P.benchmark_curves()`, `P.fit_protonation(...)`, `P.error_scenarios()`.

## Phase 2 / فاز ۲
`titration_sim.engine.phase2`: redox (Nernst, asymmetric couples via the general electron-balance solver, pH shift, activity-corrected formal potentials), precipitation (Mohr/Volhard/**Fajans**, activity-corrected Ksp, 1:1-complex solubility), EDTA (α_Y4-, **α_M with the NH3 log-beta table + hydrolysis**, gamma-corrected Kf, hardness), back titration. `titration_sim.engine.activity`: Davies / extended-Debye-Hückel / limiting-law gamma models.

## Phase 3 (instrumental) / فاز ۳ (دستگاهی)
`titration_sim.engine.phase3`: EMF layer (SCE / Ag-AgCl reference, junction offset, deterministic noise), conductometric curves (limiting λ° table, linear T correction, exact engine species) and photometry (Beer-Lambert; permanganate end-point curve). `quantity_of(model, V)` dispatches pH / E / pAg / pM so one generic session drives every family.

## Desktop UI / رابط دسکتاپ (Part 3)
```
python -m pip install -e ".[ui]"      # PyQt6 + pyqtgraph
titration-sim                          # or: python -m titration_sim   (Persian, RTL)
titration-sim --en                     # English, LTR
```
**Windows note:** run the commands from the folder that contains `pyproject.toml` (`...\riaziandshimy_project\python\titration_sim`); if `titration-sim` is not recognized, use `python -m titration_sim`. / ویندوز: دستورها را در پوشه‌ای اجرا کنید که `pyproject.toml` دارد؛ اگر `titration-sim` شناخته نشد از `python -m titration_sim` استفاده کنید.
Family selector (top of the controls): acid-base (as before), **redox / precipitation / EDTA** with their own fields and indicators (Ferroin, Mohr, Fajans, EBT, ... — the flask colour follows E, pAg or pM) and the **back-titration calculator**.
Window: controls (left) · burette + flask (middle) · quantity(V) curve (pH / E / pAg / pM), derivative, results text (right).
Buttons add 0.05 / 0.1 / 1 / 5 mL or a custom amount (Persian digits such as ۰٫۵ accepted); *Auto-run* adds titrant with a
small step near V_eq and stops at 1.4 V_eq; the flask colour follows the chosen indicator; the curve shows the faint
theoretical curve, your points, the theoretical V_eq (orange), V_eq found from *your* points (green, needs ≥ 7 points past V_eq)
and the indicator's colour band.
All logic that is not painting is in `ui/presenter.py` (Qt-free, 21 tests run anywhere); `burette_widget`, `flask_widget`,
`curve_widget`, `controls`, `main_window` are thin Qt layers.

## Test / آزمون
```
python -m pip install -e ".[dev,ui]"
pytest                       # 349 passed + 5 skipped (1 slow, 4 +SKIP UI doctests)
pytest tests/ui              # real-widget tests (PyQt6 + pyqtgraph + pytest-qt, offscreen) — 10 passed
pytest --runslow             # also runs the slow paper-scenario test (~50 s) → 319 passed
pytest --cov=titration_sim   # TOTAL 94 %; engine ≥ 82 % (spec ≥ 80 %), ui ≥ 88 % (spec ≥ 60 %)
python -m mypy titration_sim # strict → 0 issues in 25 files
python run_tests.py          # dependency-free runner (numpy only)
python run_tests.py --slow -k full
```
**Verified in my sandbox (Python 3.12, PyQt6 + pyqtgraph + pytest-qt actually installed, Qt offscreen):**
- `pytest` (full testpaths incl. doctests): **349 passed, 5 skipped** (1 slow + 4 `+SKIP` UI doctests); with `--runslow`: **350 passed, 4 skipped**
- `pytest tests/ui`: **10 passed** — the PyQt6/pyqtgraph widget tests now run against the REAL Qt API (offscreen), not stubs
- `mypy titration_sim` (strict, target 3.10): **0 issues in 25 source files**
- coverage: **TOTAL 94 %** — engine 82–100 % (spec ≥ 80 %), UI 88–100 % except `ui/app.py` 33 % (the 8-line QApplication event loop, exercised only when a real window opens)
- `run_tests.py` (numpy-only runner): 155 passed, 1 skipped
**Still needs your machine:** double-click launch from a real display (`titration-sim`), the browser/mobile run of the web lab, and re-running the suite on your interpreter. Send me the output; I will fix what it reports.

## Files / فایل‌ها
`titration_sim/engine/` — `equilibrium.py` (proton balance, Brent) · `acid_base.py` · `indicators.py` · `analysis.py` ·
`errors.py` · `stoichiometry.py` · `phase2.py` (redox/precip/EDTA/back) · `activity.py` (Davies/Debye-Hückel gammas) · `phase3.py` (EMF/conductometry/photometry) · `quantities.py` (titer/assay/LOD-LOQ/concordance/masking/carbonate/coverage) · `potentiometry.py` (paper) · `base.py` · data in `titration_sim/data/*.json`.
`titration_sim/ui/` — `presenter.py` (Qt-free logic) · `colors.py` · `strings.py` (fa/en) · `burette_widget.py` · `flask_widget.py` · `curve_widget.py` · `controls.py` · `main_window.py` · `app.py`. Launcher: `__main__.py` (`python -m titration_sim`).
`docs/MATH.md` formulas · `docs/CONTRADICTIONS.md` conflicts between sources and decisions · `docs/ROADMAP.md` phases 2–3.

## Scope notes / نکات دامنه
All families (acid–base, redox, precipitation incl. Fajans, EDTA, back titration) and the instrumental layers
(potentiometry/conductometry/photometry) **are coded and tested**. Remaining documented limits (MATH.md end):
Nernstian electrode assumption, Davies-level activity model (no SIT/Pitzer), non-1:1 analyte salts, textbook constants
— check pKa/Ksp/λ/ε against your own tables. The ROADMAP «future» row is optional enhancement, not a defect.
