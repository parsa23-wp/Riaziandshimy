# MATH.md — formulas used by the engine / فرمول‌های موتور

Section numbers match the `[MATH.md section X]` tags in the docstrings.
Sources: **S1** = "second answer" in the 445-page chat export; **S2** = prompt-1 (101 topics);
**P** = Berto et al., *Anal. Chim. Acta* 1303 (2024) 342476 (open access, CC BY).

## 0 Architecture
`state_after = engine.titrate(model, state_before, dV)` — pure function, the UI never holds chemistry state (S1 §0).

## 1 Basics
1.1 `n = m/M`, `C = n/V_solution`. 1.2 `a·C_A·V_A / … `: `C_A = a C_B V_B / (b V_A)` for `aA + bB`; `C1V1=C2V2` only for a=b=1.
1.3 Activity `a_i = γ_i [i]`; Debye–Hückel `log γ = −A z² √I/(1+B√I)`, A = 0.51, B = 1 (P, Eq. 9, comparison setting); Davies `−A z²(√I/(1+√I) − 0.3 I)`.

## 2 Equilibria (exact, no region shortcuts)
Proton balance for ligands *l* with cumulative protonation constants β_li (reference = fully deprotonated form):

    T_H = [H⁺] − Kw/[H⁺] + Σ_l C_l · Σ_i i β_li hⁱ / (1 + Σ_i β_li hⁱ)

* 2.2 strong acid (no ligand): `h = (C + √(C²+4Kw))/2`.
* 2.4 weak acid: identical to the cubic `h³ + Ka h² − (Ka C + Kw) h − Ka Kw = 0` (checked by `numpy.roots` in the tests).
* 2.5 polyprotic: same balance with n steps; α-fractions `α_i = βᵢhⁱ/(1+Σβhⁱ)`. Solved on log₁₀h with Brent (monotone ⇒ unique root; no Newton divergence for n = 6).
* 2.6 `Ka·Kb = Kw`, `pKw(T)` — see §12.

## 3 Curves
`T_H` at titrant volume V (volumes in mL, V_T = V0 + V):
analyte acid: strong `+C_a V0/V_T`, weak: ligand with `n_init = n` protons; titrant strong base: `−C_t V/V_T`;
weak base titrant: ligand (β = 10^pKa(BH⁺)) with no initial protons; weak-acid titrant: ligand and `+C_t V/V_T`.
* 3.1 strong/strong: continuous formula, `pH_eq = pKw/2` (6.63 at 50 °C, not 7).
* 3.2 weak acid/strong base: Henderson–Hasselbalch is only displayed as an *approximation* (error reported).
* 3.3 jump criterion: `ΔpH = |pH(1.001 Veq) − pH(0.999 Veq)| ≥ 2` for a visual indicator.
* 3.5 weak/weak: `pH_eq ≈ ½(pKa + pKa(BH⁺))`, jump ≪ 1 ⇒ use potentiometry.

## 4 Indicators
`[In⁻]/[HIn] = 10^(pH − pK_In)`. Endpoint model: new colour visible at fraction α_vis (0.5 ⇒ pH = pK_In; ratio 10:1 ⇒ pK_In ± 0.954…1). Optional absorbance threshold (|ΔA| > 0.02) only with user-supplied ε. Phenolphthalein fades above pH ≈ 13.

## 5 Derivatives and equivalence
5.1 non-uniform 3-point first derivative `(h1² y₊ − h2² y₋ + (h2²−h1²) y₀)/(h1h2(h1+h2))` (= central difference on uniform grids).
5.2 second derivative `2[(y₊−y₀)/h2 − (y₀−y₋)/h1]/(h1+h2)`.
5.3 Gran: strong acid `(V0+V)[H⁺] = Ca V0 − Ct V`; weak acid **`V·[H⁺] = Ka (Veq − V)`** (corrected, see CONTRADICTIONS #3); bases with `[OH⁻] = Kw/[H⁺]`.
5.4 `V_eq` = weighted mean (0.4 first-derivative, 0.4 second-derivative, 0.2 Gran).

## 10 Statistics and uncertainty
Mean, SD (n−1), RSD, SE = s/√n, CI = x̄ ± t(α/2, n−1)·SE (t from the incomplete beta function, tested against tables),
Dixon Q (95 %, n = 3…10, table values taken from the literature) and Grubbs (exact critical value from t).
10.6 `(u_C/C)² = (u_CB/CB)² + (u_VB/VB)² + (u_VA/VA)²`; 10.7 burette `u_V = √(2u_read² + u_cal²)`.

## 11 Error simulation
Burette factor, titrant factor F, CO₂ titer loss, blank, evaporation (moles conserved), overshoot, random read/drop/pipette errors, gross errors (misread 1.00 mL, wrong concentration entry, spilled sample, wrong indicator).

## 12 Temperature
12.1 `pKw = 4787.3/T + 7.1321 log₁₀T + 0.010365 T − 22.80` (T in K; 13.995 at 25 °C, 13.26 at 50 °C).
12.2 van't Hoff: `pKa(T) = pKa25 + ΔH/(R ln10)·(1/T − 1/298.15)`. 12.3 `V(T) = V25(1 + 2.1·10⁻⁴ (T − 25))` (optional).

## 13 Numerics
13.3 Brent on log₁₀h. 13.4 adaptive step: bisect every interval with |ΔpH| > 0.05 ⇒ grid is non-uniform ⇒ use §5.1 non-uniform formulas.

## 14 Potentiometry (paper)
Eq. 10: `E = E′₀ + S log[H⁺] + j_A[H⁺] + j_B[OH⁻]`.
Eqs. 1–4: `U = √(Σ wₙ(E_exp − E_calc)²/(N−p))`, `w = 1/s²`, `s² = s_E² + (∂E/∂V)² s_V²` (5-point cubic slope).
Eq. 8: `β^T = β·γ_C^c γ_D^d / (γ_A^a γ_B^b)`. Stepwise uncertainty: `s²(logK) = s²(logβ_n) + s²(logβ_{n−1}) − 2cov`.
Benchmark: hexaprotic acid logK = 10, 8, 6, 4, 3, 2; pKw = 13.77 (I = 0.1 M); E′₀ = 405 mV, S = −59.16 mV, j_A = −64; 6 curves (2, 4, 5 mM × 2); s_E = 0.15 mV, s_V = 0.005 mL.

## 15 UI behaviour (Part 3)
15.1 Burette: zero at the top, reading = V added, remaining fraction `1 - V/capacity`; delivery is clipped to the remaining volume.
15.2 Auto-run step (analyst knows nominal V_eq): `|V - V_eq|/V_eq < 1 %` → 0.04 mL, `< 5 %` → 0.1 mL, `< 20 %` → 0.4 mL, else 1 mL (each scaled by V_eq/25 mL).
15.3 Indicator colour: `colour = (1 - f) acid + f base`, `f = 1/(1 + 10^(pK_In - pH))` (section 4); phenolphthalein returns to colourless above `fade_ph`.
15.4 Result from the collected points: `C = C_t V_eq,found / V_a` (1:1, first equivalence), `V_eq,found` = `find_equivalence` (5.4), relative error `(C - C_true)/C_true`.

## 16 Phase 2 (module `engine/phase2.py`)
16.1 Redox, f = V/V_eq, V_eq = n1 C1 V1/(n2 C2): f<1 `E = E1 + (s/n1) log(f/(1-f))`; f>1 `E = E2 + (s/n2) log(f-1)`; f=1 `E=(n1E1+n2E2)/(n1+n2)`; s = 2.303RT/F; optional `-s m/n pH` shift of formal potentials.
16.2 Precipitation AgX: with precipitate `[Ag+] = (d + sqrt(d^2+4Ksp))/2`, d = Ag_tot - X_tot (exact, covers all regions); Mohr endpoint `[Ag+] = sqrt(Ksp,Ag2CrO4/[CrO4])`; Volhard `n_X = C_Ag V_Ag - C_SCN V_SCN`.
16.3 EDTA: `alpha_Y4 = 1/(1+sum beta_i h^i)`, `K' = alpha Kf`; mass balance `K' M^2 + (1+K'(Y_tot-M_tot)) M - M_tot = 0` (exact, 1:1); hardness `mg/L CaCO3 = C_Y V_Y/V_s * 100.09 * 1000`.
16.4 Back titration: `n_A = (C_R V_R - C_T V_T/ratio_T/R)/ratio_R/A`.
## 16.5 Activity coefficients (module `engine/activity.py`, v0.8)
Davies (default, I <= 0.5): `log gamma = -A z^2 (sqrt(I)/(1+sqrt(I)) - 0.3 I)`; extended DH: `-A z^2 sqrt(I)/(1+1.5 sqrt(I))`; limiting law; `A = 0.509` at 25 C (linear T interpolation). Ionic strength `I = 0.5 sum c_i z_i^2`. Neutral species gamma = 1.
* Redox formal potential: `E' = E + (s/n)(a log gamma_Ox - b log gamma_Red)` (charges `z1_ox...z2_red` optional).
* Precipitation: `Ksp' = Ksp/(gamma+ gamma-)` (charges ±1).
* Complexometry: `Kf' = Kf gamma_M gamma_Y4- / gamma_MY`, `z_MY = z_M - 4`.

## 16.6 Asymmetric redox couples (exact, v0.8)
Couples are written `a Ox + n e <=> b Red` (`a`,`b` = exponents of the Nernst ratio, defaults 1). Metal-atom balance `T = b[Ox] + a[Red]` (derived atom counts; explicit `atoms*_ox/red` for multi-atom couples such as S4O6 2-). Electron balance at ANY volume: `n1[Ox1]/a1 = n2[Red2]/b2`. Given E, each couple splits by bisection on `[Red]`; E(V) follows from the electron balance (monotone). Equivalence volume (exact stoichiometry): `V_eq = n1 c_red1 b2 c_red2 C1 V1 / (a1 c_ox1 n2 c_ox2 C2)` = `n1 C1 V1/(n2 C2)` for symmetric couples.
Verification: general solver reproduces the symmetric closed form to < 3e-5 V in the jump (the difference is the `[Ox2]` leak the closed form neglects) and < 1e-9 V elsewhere; dichromate E_eq matches the analytic expression `(E1+6E2)/7 + (s/7) log([Fe3+]/(6[Cr3+]^2))` = 1.262627 V (0.1 M Fe2+, 50 mL, C_t = 1/60 M, pH 0) to 1e-6 V.

## 16.7 Precipitation complexes (proof) and Fajans
1:1 complexes (AgCl aq, AgCl2-) contribute one Ag and one X each, so they cancel in the difference balance `[Ag+] - [X-] = Ag_tot - X_tot`; with `[Ag+][X-] = Ksp'` the free-concentration quadratic is EXACT unchanged. Only the total dissolved silver changes: `Ag_diss = [Ag+](1 + beta1[X-] + beta2[X-]^2)` (`log beta1 = 3.31`, `log beta2 = 5.25`, `precip_total_dissolved_silver`).
Fajans: the anionic dye adsorbs when the AgX surface turns positive, i.e. when `[Ag+]` first exceeds the adsorption threshold (default `1.2 sqrt(Ksp')`, parameterizable). Endpoint error for realistic concentrations < 0.2 %; dye needs `pH > pKa` (fluorescein 6.4, dichlorofluorescein 4.0, eosin 2.5 approx).

## 16.8 EDTA metal side reactions
`alpha_M = 1/(1 + sum_l beta_L,l [L]^l + sum_j beta_OH,j [OH-]^j)`, `[OH-] = 10^(pH-14)`; built-in NH3 cumulative log beta (25 C): Ag (3.31, 7.21), Cu (4.31...13.32), Zn (2.37...9.46), Ni (2.80...8.74), Cd, Co2. Effective constant `K'' = Kf alpha_Y4- alpha_M (gamma factor)`. Example: Zn, pH 10, [NH3] = 0.1 M: log alpha_M = -5.49, log K'' = 10.53 (vs 16.50 absolute).

## 17 Instrumental layers (module `engine/phase3.py`, phase 3)
17.1 Potentiometry: `EMF = (E_indicator - E_ref)·1000 + E_junction` (mV); reference electrodes vs NHE at 25 C: SCE 0.244, Ag/AgCl sat KCl 0.199, Ag/AgCl 3 M KCl 0.210. Gaussian meter noise (deterministic seed) for end-point practice.
17.2 Conductometry: `kappa = 1000 · sum lambda_i c_i` (µS/cm, c in mol/L, limiting lambda° at 25 C: H+ 349.65, OH- 198.0, Na+ 50.10, Cl- 76.35, acetate 40.9, ...), temperature `kappa(T) = kappa(25)(1 + 0.019(T-25))`. APPROXIMATIONS: no Onsager correction (a few % high at 0.1 M); all charged forms of a polyprotic ion share one lambda°. Species from the exact engine state (H+/OH-, analyte forms with charge i-n, spectators Na+/Cl-/X-/BH+).
17.3 Photometry: Beer-Lambert `A = l sum eps_i c_i`; permanganate convenience curve uses the exact post-equilibrium excess `[MnO4-]` (eps 525 nm = 2400 L mol^-1 cm^-1) — A stays at the equilibrium-leak level before the end point and rises linearly after it.
17.4 Family dispatcher `quantity_of(model, V)`: pH (acid-base), E/V (redox), pAg (precip), pM (EDTA) — one plot/derivative/end-point machinery for all families (desktop UI + web).

## 18 Quantities, non-aqueous, masking, sufficiency (module `engine/quantities.py`, v0.9)
18.1 Titer: `titer_mg_per_ml = C_t · (a/b) · MW` (mg analyte per mL titrant; a/b = moles analyte per mole titrant).
18.2 Assay: `% = n_analyte·MW/m_sample·100`, `n_analyte = C_t V_t (a/b) (V_flask/V_aliquot)`; the flask/aliquot factor scales a dissolved-and-diluted sample. `nonaqueous_assay_percent` is the same stoichiometry for a direct non-aqueous titration (titrant standardized in the SAME solvent).
18.3 Standardization: `hclo4_from_khp = m·purity/(204.22·V)` (KHP, monoprotic); `factor_f = C_actual/C_nominal`.
18.4 Concordance: `spread = max−min <= max(0.05 mL, 0.002·|mean|)` (+1e-9 float guard); dataclass `ConcordanceResult`.
18.5 LOD/LOQ: ICH `LOD = 3.3σ/S`, classic `3σ/S`, always `LOQ = 10σ/S` (style reported; no single universal definition exists).
18.6 Buffer capacity: `β = 2.303(c·Ka·h/(Ka+h)² + h + Kw/h)`, maximal at h = Ka (pH = pKa) — numeric companion to the analytic form.
18.7 pKa from half-equivalence: linear interpolation of pH at `(V_eq,k−1 + V_eq,k)/2`; exact for weak/strong, approximate for weak/weak.
18.8 Carbonate by two end points (Berto 2024 §2.6(ii)): `n_CO3 = (V2−V1)c`, `n_OH = (2V1−V2)c`; validity requires `2V1 ≥ V2`.
18.9 EDTA masking (KCN): `log K'' = log Kf + log α_Y4- + log α_M`, titratable iff `log(K''·C) >= 6` (sharp-end-point rule). Free ligand `[L] = c·Ka/(Ka+[H+])` with HCN pKa 9.21. Built-in log Kf (Ca 10.65, Mg 8.79, Zn 16.50, Cd 16.46, Cu 18.80, Ni 18.62, Pb 18.04, Mn 13.87, Fe2+ 14.32, Co 16.31, Hg 21.70, Fe3+ 25.10) and CN cumulative betas (Zn4 19.6, Cd4 18.9, Cu4 30.5 as Cu(I), Ni4 31.3, Ag2 20.5, Hg4 41.5). Metals absent from the ligand table get α_M = 1. Example verified in both engines: pH 10, 0.01 M CN⁻ → Ca/Mg/Pb/Mn titratable, Zn/Cd/Cu/Ni/Hg masked.
18.10 Berto data-sufficiency guideline: `species_coverage_check` counts, per species, the points with 20 % <= alpha <= 80 %; "adequate" = at least 10 per species (paper: 10–15 points in the 20–80 % band per species; high/low constants like logK1/logK6 are naturally the least covered).
18.11 `median` added to `engine/analysis.py`; air bubble (`bubble_ml`, fixed positive reading bias) and parallax (`sigma_parallax_ml`, per-reading normal, net sqrt(2)) added to `ErrorConfig` + two ready-made scenarios.

Remaining limits / محدودیت‌های باقی‌مانده: redox electrode kinetics (Nernstian assumption), OH-complexation of metals only through user-supplied beta values, non-1:1 analyte salts (Ag2CrO4), activity models beyond Davies/extended DH (no SIT/Pitzer), non-aqueous modeled at the calculation level only (no solvent activity-coefficient theory). pKa/logKf/Ksp/log beta(CN)/lambda/eps/KHP MW/HCN pKa are textbook values — check against your tables.

## 19 Extended families (module `engine/extended.py`, v1.0)
19.1 Non-aqueous: the aqueous proton balance with `pKw -> pKs(solvent)`; `solve_log_h` widens its lower bracket to `log10(Ks) - 4` (pKs up to ~35). `pKa` must be the *apparent* constant in that solvent; ion pairing and homoconjugation are not modelled. Verified pKs: acetic acid 14.45, methanol 16.7, ethanol 19.5; acetonitrile and DMSO (33.3) are commonly quoted but NOT cross-checked (flagged `verified=False`).
19.2 Amperometry: `i = 1000 [kA nA_left + kT nT_excess + kP xi] / V_tot + i_res` (k in uA per mmol/L); dilution correction multiplies by `(V0+V)/V0` so both branches are straight; end point = two-line intersection (points within +-10 % of V_eq excluded).
19.3 Biamperometry (dead stop): `i = k min([Ox],[Red])`.
19.4 Thermometry (adiabatic): `dT = (-dH xi/Cp + dT_titrant V)/(V0+V)`, `xi` in mmol, `dH` in kJ/mol; neutralization 5 mmol x 55.8 J/mmol / (4.18 x 100 mL) = 0.667 K.
19.5 Karl Fischer: `F = m_water/(V-V_blank)` (tartrate dihydrate 15.66 % water); `mg H2O = F (V-V_b) - drift t/1000`; coulometric: `m = Q M/(2F)`, i.e. 10.71 C per mg.
19.6 Coulometry (Faraday): `n = eta i t/(n_e F)`; bias from assuming eta = 1 is `(1-eta) x 100 %`.
19.7 Standardization: exact stoichiometry table `PRIMARY_STANDARDS` (KHP/NaOH 1:1, Na2CO3/HCl 1:2, Na2C2O4/KMnO4 5:2, KIO3/thiosulfate 1:6, CaCO3/EDTA, NaCl/AgNO3).
19.8 GUM: `u_c^2 = sum (c_i u_i)^2`, `c_i` by central difference; Welch-Satterthwaite `nu_eff = u_c^4 / sum((c_i u_i)^4/nu_i)`; k from the 95.45 % t table (nearest *lower* tabulated dof, conservative). Type B: rectangular `a/sqrt(3)`, triangular `a/sqrt(6)`. No correlations, no Monte Carlo.
19.9 EDTA pKa (corrected in v1.0): 1.99, 2.67, 6.16, 10.26 (K = 1.02e-2, 2.14e-3, 6.92e-7, 5.50e-11) reproduce the tabulated alpha_Y4-: 2.5e-11 (pH 3), 5.4e-3 (pH 8), 0.35 (pH 10), 0.98 (pH 12). The previous rounded set (2.0, 2.7, 6.2, 10.3) gave 0.334 at pH 10.

