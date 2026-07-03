I have enough to write the briefing. One key correction confirmed: the RTI engine code is public (real repo name `RTIInternational/diabetes-simbackend-only`, unlicensed), the Hoerger 2023 equations are published, but it models established T2D only — the prediabetes/vitamin-D front-end is Briody-specific. Here is the briefing.

---

# BUILD-vs-BORROW Briefing: Re-implementing the CDC/RTI Diabetes Microsimulation

## Bottom line

**Borrow the engine, don't rebuild it from scratch — but plan the strategic version in vivarium.** The core engine you are trying to reproduce is *not* fully proprietary: the CDC-RTI Type 2 Diabetes Microsimulation (Hoerger et al. 2023) has (a) all equations and coefficients published in the paper's appendices and (b) a **public Python reference implementation on GitHub**. The genuinely proprietary/unpublished part is the thin *prediabetes→onset + vitamin-D intervention* front-end that Briody 2026 bolted on. So the fastest faithful replication is: use the public RTI code + published equations as your reference, and re-implement the prevention front-end from Briody's methods. For your GBD-rebase and other-locations goals, the right long-term home is **IHME's own `vivarium` / `vivarium_public_health`**, which is purpose-built for this exact class of model. **LASER is the wrong tool** (spatial infectious-disease ABM).

---

## 1. IHME open-source microsimulation frameworks

### vivarium + vivarium_public_health — **strong fit**

- **vivarium** (`ihmeuw/vivarium`) is a general-purpose, **discrete-time, individual-based (microsimulation) framework** on the scientific-Python stack. Time step is configurable (e.g., 365 days → your annual step). Repo: https://github.com/ihmeuw/vivarium · PyPI: https://pypi.org/project/vivarium/ · Docs: https://vivarium.readthedocs.io
- **vivarium_public_health** (`ihmeuw/vivarium_public_health`, **BSD-3-Clause** — permissive, commercial-friendly) provides exactly the component library this model needs:
  - **Disease models**: state-machine disease components (SI / SIS / SIR and custom multi-state) with `SusceptibleState`/`DiseaseState`/`TransientDiseaseState`, per-state excess mortality and disability weights, and configurable transition hazards. A `normoglycemia → prediabetes → diabetes → complications → death` progression is expressible as a custom multi-state machine.
  - **Risk**: `RiskExposure` (continuous/categorical exposures, incl. correlated exposures via propensity/correlation) and `RiskEffect` (relative-risk application of exposures onto disease hazards) — maps onto your time-varying BMI/HbA1c/BP/lipids/eGFR driving complication hazards.
  - **Treatment / intervention**: `Treatment`, `Intervention`, `InterventionEffect`, `AbsoluteShift`, therapeutic scale-up — the natural place for the vitamin-D hazard ratio on diabetes onset.
  - **Observers**: results/mortality/disease observers with stratification; cost and QALY/DALY accumulation are done as observer components.
  - Docs: https://vivarium.readthedocs.io/projects/vivarium-public-health/en/latest/
- **Important repo-status note:** as of **18 June 2026** `vivarium_public_health` was **archived and migrated into the `vivarium-suite` monorepo**; import path changed from `vivarium_public_health` to `vivarium.public_health`. Build against the new suite, not the archived repo.
- **Caveats / effort:** vivarium's "batteries" assume GBD-style cause/risk exposure data. Your model is driven by **trial-derived multivariable Weibull/Gompertz hazard equations** (ACCORD/Look AHEAD), so the 17 complication + 3 mortality + risk-progression equations become **custom components** you write yourself. There's a real learning curve, but this is precisely what IHME uses vivarium for — it is the strategic fit for goals 2 (GBD re-base) and 3 (other locations), because it natively ingests GBD exposure/RR data and is designed for multi-location reuse.

### LASER / laser-core — **not appropriate here**

- LASER ("Light Agent Spatial modeling for ERadication") is an IDM/IHME-adjacent framework for **large-scale spatial, transmission-driven infectious-disease ABMs** (measles, malaria eradication), with node networks, migration/mixing, and numba-accelerated structured arrays. MIT-licensed. Repos: https://github.com/InstituteforDiseaseModeling/laser · https://github.com/laser-base/laser-core (PyPI `laser-core`).
- Your problem has **no transmission and no spatial interaction** — it's independent per-individual chronic-disease hazard trajectories. LASER's core machinery (contagion, spatial connectivity) buys you nothing, and you'd be fighting its assumptions. Its raw performance could run 100×10,000 fast, but `numpy`/`numba` alone gets you that without the framework. **Skip LASER.**

---

## 2. Existing diabetes models — open vs. licensed, equations published?

| Model | Open code? | Equations published? | Notes / URL |
|---|---|---|---|
| **CDC-RTI T2D Microsimulation (Hoerger 2023)** — *the engine you're replicating* | **Yes, public Python** (no license file → all-rights-reserved by default; contact RTI to reuse) | **Yes** — 17 Weibull complication eqs, 3 mortality eqs (Gompertz/logistic), risk-factor progression, Optum cost eqs, HUI-3 utility, in appendices | Repo: https://github.com/RTIInternational/diabetes-simbackend-only · Paper (Value in Health 26(9):1372-1380): https://pmc.ncbi.nlm.nih.gov/articles/PMC11017333/ · https://www.mthooddiabeteschallenge.com/cdcrti |
| **CDC-RTI Diabetes CE Model (older, Markov; Hoerger 2004+)** — used for DPP/prediabetes screening CE | No | Partially | The prevention-oriented predecessor; validation: https://www.rti.org/rti-press-publication/diabetes-cost-effectiveness |
| **UKPDS Outcomes Model 2 (OM2)** | Licensed software (Oxford/OUI) | **Yes** — 13-equation illness-death model, Hayes 2013 *Diabetologia* (UKPDS 82); risk-factor progression eqs Leal 2021 (UKPDS 90) | https://link.springer.com/article/10.1007/s00125-013-2940-y · https://www.mthooddiabeteschallenge.com/copy-of-ukpds-1 |
| **Sheffield T2D model (ScHARR)** | No | Partially (5 submodels: CHD, stroke, nephropathy, retinopathy, neuropathy) | University of Sheffield |
| **MICADO** | No | **Yes** — Netherlands population model, van der Heijden 2015 | https://pubmed.ncbi.nlm.nih.gov/26010494/ · transferability: https://pmc.ncbi.nlm.nih.gov/articles/PMC9156453/ |
| **Michigan Model for Diabetes** | No | **Yes** — CHD submodel Development/Validation 2015 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4696433/ |
| **IQVIA (IMS) CORE Diabetes Model** | **No — proprietary, licensed** | Structure published; coefficients not fully | Interconnected Markov sub-models; Mount Hood validated: https://www.tandfonline.com/doi/full/10.1080/13696998.2023.2240957 |
| **PROSIT** | **Yes — genuinely open source** | Yes (transparent) | Markov models (MI, stroke, retinopathy, nephropathy, foot, hypoglycemia) in OpenOffice Calc, downloadable: https://pubmed.ncbi.nlm.nih.gov/27350481/ |

**Cross-model validation hub:** the **Mount Hood Diabetes Challenge** (https://www.mthooddiabeteschallenge.com/) publishes standardized reference scenarios and reports — your primary external validation target.

**On Briody 2026 specifically:** "The cost-effectiveness of vitamin D supplementation for the prevention of type 2 diabetes in the United States: a microsimulation modelling study," *Lancet Regional Health – Americas* 61:101543 (Briody, Pittas, Zhang, Shao, Gregg), https://doi.org/10.1016/j.lana.2026.101543. Ping Zhang (CDC) and the design point to the **CDC-RTI microsim engine** as the base. The 15% RRR / vitamin-D effect derives from the D2d trial (Pittas, NEJM 2019, https://www.nejm.org/doi/full/10.1056/NEJMoa1900906) and IPD meta-analysis (Pittas, *Ann Intern Med* 2023, HR≈0.85, https://www.acpjournals.org/doi/10.7326/M22-3018).

---

## 3. Effort to re-implement the CDC/RTI model faithfully in Python

**Because the reference code exists, this is a port-and-extend job, not a clean-room build.**

- **Borrow-and-port path (recommended): ~6–12 weeks / 1.5–3 months for one experienced health-economic modeler.** Port/clean the 17 complication + 3 mortality + risk-progression + cost + HUI-3 utility components from the public RTI code (cross-checked against the Hoerger 2023 appendices), then add: (a) the correlated synthetic-cohort initializer from NHANES, (b) the **prediabetes→diabetes onset front-end + vitamin-D HR + 85% adherence** (Briody-specific), (c) discounting, ICER, NMB, and the 100×10,000 Monte Carlo harness.
- **Clean-room from published equations only (ignore the code): ~4–6 months**, materially higher risk on the underspecified pieces below.

**Main risks (ranked):**
1. **The prevention front-end is the least-documented part.** The prediabetes→onset module + how the vitamin-D HR is applied (to a discrete onset hazard vs. continuous HbA1c) is Briody's novel contribution and may exist only in that paper's methods. This — not the complication engine — is where replication will actually be hard.
2. **Correlation structure of the synthetic population.** Multivariate sampling of correlated baseline risk factors from NHANES (copula/Cholesky details, and which NHANES cycles/weights) is routinely underspecified. *Note your CLAUDE.md rule: apply NHANES survey weights (MEC weight, e.g. `WTMECPRP` for 2017–2020) when building the cohort.*
3. **Risk-factor trajectory equations & update order.** The paper states risk factors update in random order with smoking last; exact progression coefficients, capping/bounds, and feedback loops must match or complication incidence drifts.
4. **Unpublished/rounded coefficients.** Appendix coefficients may be rounded or omit interaction terms present in the code.
5. **Validation targets.** You need concrete numbers to hit: Mount Hood reference outputs, ACCORD/Look AHEAD internal-validation figures, and Briody's reported ICER/NMB. Without these you can't prove faithfulness.

**Licensing caveat:** the RTI repo has **no LICENSE file** → default all-rights-reserved. You may read and learn from it freely; for reuse/redistribution/derivative release, get written permission from RTI/Hoerger. (Contrast: `vivarium_public_health` is BSD-3 and PROSIT is open.)

---

## 4. Recommended architecture — full microsim vs. Markov cohort

**Trade-offs against your three goals:**

| Approach | Effort | Goal 1: Replicate Briody | Goal 2: GBD re-base | Goal 3: Other locations |
|---|---|---|---|---|
| **A. Port public RTI engine (+prevention front-end)** | Low–med (6–12 wk) | **Best fidelity** — same equations/code | Weak — US-trial equations, not GBD-native | Weak — needs re-derived equations per location |
| **B. Rebuild in `vivarium_public_health`** | High (3–5 mo) | Good (once equations ported as components) | **Best** — ingests GBD exposure/RR, multi-location by design | **Best** — one model, many locations |
| **C. Markov cohort state-transition** (`normo→prediab→diabetes→complications→death`) | Very low (2–4 wk) | Approximate only | **Good** — easy to re-base transition probs to GBD | **Good** — swap location-specific rates |

**Why not just do the simple Markov cohort?** It's the right tool for fast scenario scans and quick GBD re-basing, and it's transparent. But an aggregate cohort model **cannot reproduce Briody's ICER faithfully**: it loses individual-level heterogeneity, the **correlation** among risk factors, and continuous risk-factor trajectories — and it cannot represent the vitamin-D effect the way Briody does (a hazard modifier on onset conditioned on continuous HbA1c). Use it as a *cross-check*, not the primary replication.

**Recommended plan — two-tier / hybrid:**
1. **Now, for Goal 1 (faithful replication):** port the **public RTI engine** as your reference implementation, add the Briody prevention front-end, and validate against Briody's published ICER/NMB and Mount Hood scenarios. Fastest defensible path to "we reproduced it."
2. **Next, for Goals 2 & 3 (GBD re-base + other locations):** re-implement the same equations as **`vivarium`/`vivarium_public_health` components** (new `vivarium.public_health` import path). This is the strategic home — IHME's framework gives you GBD exposure/RR ingestion, correlated risk-factor exposures, intervention components, and multi-location reuse out of the box.
3. **Optional companion:** a lightweight **Markov cohort** version for rapid sensitivity scans and sanity-checking the microsim — cheap insurance, borrow structure from PROSIT (openly licensed).

**One-line recommendation:** *Borrow the RTI engine to replicate Briody quickly; build the durable version in vivarium for GBD re-basing and geographic transfer; keep a Markov cohort as a fast cross-check.*

---

### Key URLs
- RTI engine (public code): https://github.com/RTIInternational/diabetes-simbackend-only · paper https://pmc.ncbi.nlm.nih.gov/articles/PMC11017333/
- Briody 2026: https://doi.org/10.1016/j.lana.2026.101543
- vivarium: https://github.com/ihmeuw/vivarium · vivarium_public_health (archived, now `vivarium-suite`): https://github.com/ihmeuw/vivarium_public_health · docs https://vivarium.readthedocs.io/projects/vivarium-public-health/en/latest/
- LASER: https://github.com/InstituteforDiseaseModeling/laser · https://github.com/laser-base/laser-core
- UKPDS OM2: https://link.springer.com/article/10.1007/s00125-013-2940-y · PROSIT (open): https://pubmed.ncbi.nlm.nih.gov/27350481/ · MICADO: https://pubmed.ncbi.nlm.nih.gov/26010494/ · Michigan: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4696433/ · IQVIA CORE: https://www.tandfonline.com/doi/full/10.1080/13696998.2023.2240957 · Mount Hood: https://www.mthooddiabeteschallenge.com/
- D2d evidence: https://www.nejm.org/doi/full/10.1056/NEJMoa1900906 · https://www.acpjournals.org/doi/10.7326/M22-3018