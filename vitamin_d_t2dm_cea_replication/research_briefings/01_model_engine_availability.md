I have everything needed. Compiling the briefing.

---

# Reproducibility Briefing: Briody et al. 2026 Vitamin-D/T2D Microsimulation and the CDC/RTI Engine

**Bottom line up front:** The paper's data-sharing statement understates what is already public. The Hoerger 2023 engine is not merely "described" — the **complete, parameterized Python engine (every complication, mortality, and risk-factor coefficient, with standard errors for PSA) is on public GitHub and I cloned it anonymously today**. The precursor CDC-RTI Markov model, by contrast, is genuinely "described but not fully specified." The Briody Mendeley deposit is **not discoverable as of 2026-07-03**.

---

## 1. The Hoerger 2023 model — architecture, equations, sources, reproducibility

**Citation:** Hoerger TJ, Hilscher R, Neuwahl S, Kaufmann MB, Shao H, Laxy M, Cheng YJ, Benoit S, Chen H, Anderson A, Craven T, Yang W, Cintina I, Staimez L, Zhang P; Look AHEAD Research Group. *A New Type 2 Diabetes Microsimulation Model to Estimate Long-Term Health Outcomes, Costs, and Cost-Effectiveness.* Value Health. 2023;26(9):1372-1380. (Per PubMed; [DOI 10.1016/j.jval.2023.05.013](https://doi.org/10.1016/j.jval.2023.05.013); open access CC BY-NC-ND; PMID 37236396; free full text [PMC11017333](https://pmc.ncbi.nlm.nih.gov/articles/PMC11017333/), CDC Stacks [cdc/154386](https://stacks.cdc.gov/view/cdc/154386)). Note co-author **Ping Zhang (CDC)** is also a Briody 2026 co-author — direct provenance link.

**Architecture (confirmed from paper + code):** Discrete-time individual microsimulation, **1-year time step**, programmed in **Python 3.8+**. Each cycle, for every simulated person: (1) run the 17 complication risk equations, each a Monte-Carlo draw (uniform 0–1 vs. the equation's annual probability); (2) run the mortality equations; (3) update risk factors and complication history for survivors; (4) advance one year until death or the time horizon. Stochastic uncertainty handled by increasing N; parameter uncertainty by **probabilistic sensitivity analysis (PSA)** drawing coefficients from their sampling distributions.

**The ~17 complication equations** are **multivariate parametric Weibull hazard models** with time-varying, time-lagged covariates. The GitHub repo confirms exactly which ones (`src/simulation/equations/t2d/complication_*.py`) — **17 clinical complications plus a composite CVD gate**:

| Renal/metabolic | Eye | Neuro/foot | Cardiovascular | Hypoglycemia |
|---|---|---|---|---|
| microalbuminuria, macroalbuminuria, eGFR<60, eGFR<30, dialysis | laser_retina (retinal photocoagulation), blindness | neuropathy, ulcer, amputation | MI, stroke, CHF, angina, revascularization (+ composite `cvd`) | hypoglycemia_any, hypoglycemia_medical |

**The mortality equation is actually 3 equations**, split by CVD status (files `mortality_*.py`): (1) **no CVD history** — Gompertz hazard on age; (2) **death in the year of an incident CVD event** — logistic; (3) **established/prior CVD, survived >1 yr** — Gompertz. All estimate **all-cause** mortality. The paper explicitly recommends **calibrating mortality for horizons >10 years** because the model under-predicts it — a reproducibility caveat any re-user must handle.

**ORIGINAL SOURCES of the equations (the model's defining feature): US data, NOT UKPDS.** This is the deliberate break from every UKPDS-Outcomes-Model-based simulator:
- **Complication, mortality, and risk-factor-progression equations:** pooled longitudinal data from **ACCORD** (+ ACCORD Follow-On) and **Look AHEAD** (+ follow-on) — two large, diverse US T2D trials, ≤13-yr follow-up.
- **Patient utility (QALY) equation:** **Health Utilities Index Mark III (HUI3)** collected in the same ACCORD + Look AHEAD populations.
- **Complication costs:** **Optum de-identified Normative Health Information** database (large US private-claims panel).
- **Default baseline population:** **NHANES 2009-2010 through 2015-2016** adults with diagnosed diabetes.
- **External validation only** (not equation derivation): ADVANCE, ASPEN, DECLARE-TIMI 58, VADT (trials) and JHS, MESA (cohorts).

**Are coefficients published enough to re-implement? YES — beyond re-implementation.** Two independent routes:
- **Journal appendices** (Word docs on PMC/CDC Stacks): `supplement-4` = **Appendix 1, risk-equation estimation & parameter estimates**; `supplement-2` = Appendix 2, modeling details; `supplement-3` = Appendix 3, validation report.
- **The public code carries every coefficient AND its standard error** (for PSA). Example — the MI Weibull equation (`complication_mi.py`), values are `[point_estimate, SE]`:
  ```
  "lambda_val":[-10.829,0.588], "shape":[-0.025,0.026], "age_entry":[0.033,0.004],
  "twd_hba1c":[0.156,0.029], "female":[-0.278,0.064], "ever_revasc":[0.630,0.066], ...
  ```
  and the non-CVD-death Gompertz (`mortality_non_cvd_death.py`): `"intercept":[-11.943,0.558], "shape":[0.090,0.005], ...`. Functional forms are explicit too: Weibull integrated hazard `exp(λ+Xβ)·t^shape`; Gompertz `(1/φ)·exp(λ+Xβ)·(exp(φ·t)−1)`. Risk-factor progression coefficients (e.g., `progression_hba1c.py`) are likewise hardcoded. Cost/disutility/multiplier values live in the scenario JSONs (`disutilities_t2d`, `costs_t2d`, `complication_multipliers_t2d`).

**Verdict: fully specified and directly executable.** This is the strongest tier of reproducibility.

---

## 2. Is the CDC/RTI engine source code obtainable? — YES, it is public

**The engine is on public GitHub:** **https://github.com/RTIInternational/diabetes-simbackend-only** (the Hoerger paper's data statement renders the slug with an extra hyphen as "diabetes-sim-backend-only," which 404s — the real slug has no hyphen between "sim" and "backend"). Public Python repo, 5 stars / 3 forks, single "initial commit," **last updated 11 Oct 2023**. **I cloned it anonymously today (2026-07-03)** with no credentials — contradicting the paper's claim that "a github account is required" (it may have been gated at publication and later opened).

**What the clone actually contains** (verified locally at `/tmp/claude-0/-home-user-ai-assisted-research/0204593c-8a54-599e-a074-6e5ddef73666/scratchpad/diabetes-simbackend-only`):
- `src/simulation/equations/t2d/` — all 17 complication + 4 mortality + 9 risk-factor-progression + 6 economics (cost/QALY) modules, **coefficients embedded**.
- `src/simulation/equations/models/` — Weibull, Gompertz, logistic, normal, beta, gamma, exponential, uniform samplers.
- `data/default-populations/` — ready-to-run input populations: **`t2d_nhanes.csv` (2,312 individuals, keyed on NHANES SEQN)**, plus `t2d_accord.csv`, `t2d_lookahead.csv`.
- `scenarios/*.json` — 14 scenario configs with the numeric cost, disutility, and multiplier parameters and intervention definitions (glycemic/BP/cholesterol/smoking).
- `src/analysis/output/` — example CE/incidence/prevalence output CSVs; `run-model.py` entrypoint; `requirements.txt` (numpy/pandas/numba, Python 3.8-3.10).

**The critical nuance that reconciles "public" vs. "proprietary":**
- It is labeled **"backend-only"** — the interactive web/UI frontend is excluded, but the **simulation engine itself is present and runnable**.
- **There is NO LICENSE file.** Source that is publicly *readable and cloneable* but carries no license is, by default, **all-rights-reserved** — copyright retained by RTI/CDC. So Briody's "proprietary…not publicly distributable" is defensible as a statement about *redistribution/reuse rights*, even though the code is *publicly visible and clonable*. These are not contradictory; the paper's framing is just misleadingly strong about accessibility.

**Access required to run it:** none beyond a web browser / `git clone` + Python. **Public technical user manual:** no standalone manual — documentation = the three journal appendices plus per-folder `README.md` files in the repo. To get the polished, supported product (frontend, updates), you would contact RTI/CDC (Ping Zhang), but that is not needed to reproduce or extend the engine.

---

## 3. Is the Briody 2026 Mendeley Data deposit live? — Not discoverable as of 2026-07-03

**Authors** (from search): Jonathan Briody, Anastassios G. Pittas, Ping Zhang, Yixue Shao, Edward Gregg. Paper DOI [10.1016/j.lana.2026.101543](https://doi.org/10.1016/j.lana.2026.101543) (PII S2667193X26001730, Lancet Reg Health Am).

**Deposit status — negative on every channel checked today:**
- **Mendeley Data** search for "Briody vitamin D diabetes" returns **no matching dataset**.
- **PubMed**: the Briody article itself is **not yet indexed** (0 hits) — consistent with a just-published 2026 article; no PMC full text yet.
- **General web search**: no Mendeley DOI/landing page surfaced.
- I could **not** retrieve the paper's verbatim data-sharing statement independently — ScienceDirect and thelancet.com both return **HTTP 403** to the fetcher. So the deposit description below relies on the statement as you summarized it, not on my own read of the paper.

**What the deposit is expected to contain vs. not** (per the stated plan): **Contains** — input tables, post-processing R/Python scripts, a deidentified NHANES-derived analytic dataset, and (likely) aggregate outputs. **Does NOT contain** — the simulation engine (the "proprietary" CDC/RTI code).

**Skeptical observation:** the deposit's value is partly redundant. A **NHANES-derived analytic population already ships publicly** in the RTI repo (`t2d_nhanes.csv`), and the engine + its input tables/scenarios are public there too. The genuinely paper-specific artifacts the deposit would add are (a) the **vitamin-D intervention effect assumptions** (D2d-trial-derived risk reduction, follow-on assumptions), (b) their **specific scenario/config files**, and (c) their **post-processing scripts** — these are the pieces you cannot reconstruct from the public repo alone. Until the deposit is live, those remain unavailable, so **full end-to-end reproduction of Briody's specific results is currently blocked**, even though the underlying engine is not.

---

## 4. The precursor CDC-RTI Diabetes Cost-Effectiveness Model — "described, not fully specified"

**What it is:** a **Markov state-transition** cohort model (not a microsimulation), with **four modules** (diabetes, diabetes screening, prediabetes, prediabetes screening). Complications modeled: **nephropathy** (microalbuminuria → macroalbuminuria → ESRD/dialysis), **neuropathy**, **retinopathy**, **CHD/CVD** (two pathways), **stroke**, and **mortality**.

**Lineage and original equation sources:** descends from the original CDC diabetes-complications cost-effectiveness framework — **Eastman et al. 1997** (Diabetes Care) and the **CDC Diabetes Cost-Effectiveness Study Group** (JAMA 1998/2002). Its transition/risk equations are drawn from **UKPDS, Framingham, DCCT, and WESDR (Wisconsin Epidemiologic Study of Diabetic Retinopathy)** — i.e., the UK/foreign-derived equations that Hoerger 2023 explicitly set out to replace with US ACCORD/Look AHEAD equations.

**Public documentation:**
- **Hoerger TJ, Segel JE, Zhang P, Sorensen SW.** *Validation of the CDC-RTI Diabetes Cost-Effectiveness Model.* RTI Press Methods Report **MR-0013-0909**, 2009 — free PDF: https://www.rti.org/rti-press-publication/validation-cdc-rti-diabetes-cost-effectiveness-model (47 internal/external validation exercises against 24 trials).
- **Mount Hood Diabetes Challenge Network** model registry page: https://www.mthooddiabeteschallenge.com/cdcrti

**Reproducibility verdict: genuinely "described but not fully specified."** The RTI report and Mount Hood page describe the module structure and validation but **do not print the equation coefficients in one place**; they point to the underlying UKPDS/Framingham/WESDR/DCCT/Eastman publications. Re-implementing the precursor therefore means **assembling equations from multiple primary sources** — materially harder and more error-prone than the 2023 microsimulation. There is **no public source repository** for the precursor engine (unlike the 2023 model).

---

## Reproducibility tiering (skeptical summary)

| Component | Status | Can you rebuild/run it? |
|---|---|---|
| **Hoerger 2023 microsim engine** | **Public source + all coefficients (w/ SEs) + input populations + scenarios on GitHub; also full param appendix in journal** | **Yes — clone and run today; or re-implement from Appendix 1.** Strongest tier. Watch the >10-yr mortality-calibration caveat. |
| Legal right to *redistribute* that engine | No license file → all-rights-reserved (RTI/CDC) | Readable/runnable, but reuse/redistribution not formally granted. This is what "proprietary" refers to. |
| **Briody 2026 vitamin-D specifics** (intervention effects, their scenarios, post-processing scripts, their NHANES analytic set) | **Mendeley deposit not live/discoverable as of 2026-07-03** | **No, not yet** — this is the actual current blocker to reproducing *their* numbers. |
| **Precursor CDC-RTI Markov model** | Described in RTI Press MR-0013-0909 + Mount Hood page; coefficients scattered across UKPDS/Framingham/WESDR/DCCT/Eastman | Partial — requires reassembling equations from primary sources; no code repo. |

**Cloned repo for inspection:** `/tmp/claude-0/-home-user-ai-assisted-research/0204593c-8a54-599e-a074-6e5ddef73666/scratchpad/diabetes-simbackend-only` (re-clonable via `git clone https://github.com/RTIInternational/diabetes-simbackend-only.git`).

**Sources:** [Hoerger 2023 PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11017333/) · [DOI 10.1016/j.jval.2023.05.013](https://doi.org/10.1016/j.jval.2023.05.013) · [CDC Stacks cdc/154386](https://stacks.cdc.gov/view/cdc/154386) · [RTI GitHub repo](https://github.com/RTIInternational/diabetes-simbackend-only) · [Briody 2026 DOI](https://doi.org/10.1016/j.lana.2026.101543) · [RTI Press validation report MR-0013-0909](https://www.rti.org/rti-press-publication/validation-cdc-rti-diabetes-cost-effectiveness-model) · [Mount Hood CDC-RTI page](https://www.mthooddiabeteschallenge.com/cdcrti) · [RTI publication page for Hoerger 2023](https://www.rti.org/publication/new-type-2-diabetes-microsimulation-model-estimate-long-term-health-outcomes-costs-and-cost). (Hoerger 2023 metadata and full text retrieved via PubMed/PMC.)

*Caveat: I could not independently read the Briody 2026 full text (Elsevier/Lancet returned HTTP 403) or confirm its verbatim data-sharing statement; the paper is not yet in PubMed/PMC. Q3's deposit-status conclusion rests on Mendeley/web/PubMed searches all returning nothing as of 2026-07-03.*