"""Real-browser test of the web tab 'غیرآبی و روش‌های ابزاری' + cross-check JS vs the Python engine.

Needs ``pip install playwright`` and ``playwright install chromium``; skipped otherwise.
Run:  python -m tests.browser.test_web_ext   (or pytest tests/browser)
"""
from __future__ import annotations

import math
import pathlib
import sys

from titration_sim.engine.acid_base import ph_at
from titration_sim.engine import extended as E

HTML = pathlib.Path(__file__).resolve().parents[4] / "titration-lab.html"


def _pw():  # pragma: no cover - environment dependent
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    return sync_playwright


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def run() -> int:  # pragma: no cover - environment dependent
    sp = _pw()
    if sp is None:
        print("playwright not installed: skipped")
        return 0
    errors: list[str] = []
    fails: list[str] = []
    with sp() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 390, "height": 844})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.goto(HTML.as_uri())
        pg.click('nav button[data-tab="ext"]')
        # ---- UI: every family runs and shows rows
        for fam in ("nonaq", "amp", "thermo", "kf", "coul", "std", "gum"):
            pg.click(f'#ex-fam button[data-fam="{fam}"]')
            pg.click("#ex-run")
            if not pg.inner_text("#ex-res").strip() or pg.query_selector("#ex-msg .err"):
                fails.append(f"UI {fam}: no result / error: {pg.inner_text('#ex-msg')}")
        # ---- numeric cross-checks JS vs Python
        js = pg.evaluate("""() => {
          const X = window.TLExt, C = window.TitrationLab.core, o = {};
          const sim = C.buildSim({kind:'acid',C:0.1,V0:25,Ct:0.1,pKa:[20],pKw:33.3,pKwAuto:false,T:25,dH:0,thermal:false,tWeak:false});
          o.nonaq = [24.975, 25, 25.025].map(v => sim.pH(v));
          const am = {cA:0.01,vA:50,cT:0.1,ratio:1,kA:10,kT:4,kP:0,res:0};
          o.amp = [0, 2, 5, 8].map(v => X.ampCurrent(am, v, true));
          const th = {cA:0.1,vA:50,cT:0.1,dH:-55.8,cp:4.18,dtT:-3,ratio:1};
          o.th = [0, 25, 50, 80].map(v => X.thermoDt(th, v, false));
          o.kf = [X.kfTiter(100,4,'sodium_tartrate_dihydrate',0), X.kfWaterPct(3,5,1000,10,3,0), X.coulKfUg(10.711)];
          o.coul = X.coulMmol(5, 123, 2, 0.998);
          o.std = Object.keys(X.STD).map(k => X.standardize(k, 0.2, 25, 0.999));
          const q = [{name:'c',value:0.1,u:0.0002},{name:'vt',value:25,u:0.03},{name:'va',value:25,u:0.03},{name:'ep',value:0,u:0.02}];
          const r = X.gumBudget((c,vt,va,ep)=>c*(vt+ep)/va, q);
          o.gum = [r.value, r.uc, r.U, r.k, ...r.contributions.map(c=>c.share)];
          const v=[1,2,3,4,7,8,9,10], y=v.map(x=>x<5?10-x:5+2*(x-5)); o.tl = X.twoLine(v,y,5,0.1);
          return o; }""")
        py_sim = E.nonaqueous_weak_acid("acetonitrile", 0.1, 25.0, 0.1, 20.0)
        ref = {
            "nonaq": [ph_at(py_sim, v) for v in (24.975, 25.0, 25.025)],
            "amp": [E.amperometric_current(E.AmperometricModel(0.01, 50.0, 0.1, 1.0, k_analyte=10.0, k_titrant=4.0), v, True) for v in (0, 2, 5, 8)],
            "th": [E.thermometric_dt(E.ThermometricModel(0.1, 50.0, 0.1, -55.8, titrant_dt_k=-3.0), v) for v in (0, 25, 50, 80)],
            "kf": [E.kf_titer_mg_per_ml(100, 4, "sodium_tartrate_dihydrate"), E.kf_water_percent(3, 5, 1000, 10, 3), E.coulometric_kf_water_ug(10.711)],
            "coul": E.coulometric_amount_mmol(5, 123, 2, 0.998),
            "std": [E.standardize_titrant(k, 0.2, 25, 0.999) for k in E.PRIMARY_STANDARDS],
        }
        gq = [E.Quantity("c", 0.1, 0.0002), E.Quantity("vt", 25.0, 0.03), E.Quantity("va", 25.0, 0.03), E.Quantity("ep", 0.0, 0.02)]
        gr = E.gum_budget(lambda c, vt, va, ep: c * (vt + ep) / va, gq)
        ref["gum"] = [gr.value, gr.u_c, gr.u_expanded, gr.k, *[c[3] for c in gr.contributions]]
        for key, rv in ref.items():
            jv = js[key]
            jl = jv if isinstance(jv, list) else [jv]
            rl = rv if isinstance(rv, list) else [rv]
            for i, (x, y) in enumerate(zip(jl, rl)):
                if not _close(float(x), float(y), 1e-6):
                    fails.append(f"JS vs Python {key}[{i}]: {x} vs {y}")
        if not _close(js["tl"], 5.0, 1e-9):
            fails.append(f"twoLine {js['tl']}")
        pg.screenshot(path=str(pathlib.Path("/tmp/ext_tab.png")))
        b.close()
    for f in fails + errors:
        print("FAIL/ERR:", f)
    print("browser ext test:", "OK" if not fails and not errors else "FAILED")
    return 1 if fails or errors else 0


def test_web_ext_tab() -> None:
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
