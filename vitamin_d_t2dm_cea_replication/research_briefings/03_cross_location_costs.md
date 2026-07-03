I have enough to synthesize. Returning the briefing.

---

# Costing Briefing: Adapting a US Diabetes CEA Microsimulation to Other Countries

Scope note: Briody 2026's cost engine is entirely US-specific — Yang 2020 (<65 complication costs) and Wang 2022 (≥65 Medicare), 2024 USD, health-system perspective, $100k/QALY WTP. Porting it requires three swappable inputs: (a) location-specific health-system costs of diabetes complications, (b) a currency/price-year/PPP transfer method, and (c) a location-appropriate WTP threshold. Below is where each can be sourced, with candor about what is genuinely usable.

---

## 1. IHME's own costing work

**Bottom line: IHME has one deep disease-by-condition costing product — DEX — and it is US-only. Its global financing products do not go to the disease-complication level you need.**

### Disease Expenditure Project (DEX)
- **What it is:** US personal health-care spending decomposed simultaneously by condition, county, age/sex, payer, and type of care. Latest public releases cover **3,110 US counties, ~148 health conditions, 38 age–sex groups, 4 payers (Medicare/Medicaid/private/OOP), 7 types of care (ambulatory, dental, ED, home health, inpatient, nursing facility, retail Rx), 2010–2019** (earlier series 1996–2013). Built from ~40 billion claims + ~1 billion facility records.
- **Geographic scope: United States only.** There is no non-US DEX.
- **Usability for non-US locations: not directly usable.** It is, however, the closest methodological analogue to what Yang/Wang provide and could serve as the *US baseline* in a cost-transfer exercise (see §4). Diabetes is one of its conditions, and it separates type of care, which loosely maps to complication categories.
- Links: [DEX landing](http://www.healthdata.org/dex) · [DEX project overview](http://www.healthdata.org/dex/project-overview) · [GHDx: US spending by cause & county 2010–2019](https://ghdx.healthdata.org/record/ihme-data/us-health-care-spending-cause-county-2010-2019) · [GHDx: US spending variation 2019](https://ghdx.healthdata.org/record/ihme-data/us-health-care-spending-variation-cause-county-2019)

### Global Health Spending / Financing Global Health (FGH)
- **What it is:** All-country health spending, 1995–2023 (latest release 11 Jun 2026), disaggregated by **financing source** (government, out-of-pocket, prepaid private) plus development assistance for health (DAH). Country-level for ~195 countries.
- **Scope: global**, but the disaggregation is by *who pays*, not *what disease* — except for a handful of DAH-tracked causes (HIV, TB, malaria, RMNCH, etc.). **There is no diabetes-complication cost decomposition.**
- **Usability:** Useful only as a macro *scaling denominator* — e.g., total health spending per capita, government health spending per capita — to inform cost-transfer ratios or plausibility checks, not to populate complication costs.
- Links: [Financing Global Health viz (vizhub.healthdata.org/fgh)](https://www.healthdata.org/data-tools-practices/interactive-visuals/financing-global-health) · [GHDx: Global Health Spending 1995–2023](https://ghdx.healthdata.org/record/ihme-data/global-health-spending-1995-2023-20260611) · [GHDx: Global Expected Health Spending 2023–2050](https://ghdx.healthdata.org/record/ihme-data/global-expected-health-spending-2023-2050) · [Health financing hub](https://www.healthdata.org/health-financing)

**Verdict on IHME:** Great for the US (DEX) and for macro spending envelopes (FGH); **no off-the-shelf non-US complication unit costs.** GBD gives you epidemiology, not money — confirmed.

---

## 2. WHO-CHOICE, WHO GHED, and cost-of-illness literature

### WHO-CHOICE (CHOosing Interventions that are Cost-Effective)
- **What it provides:** Country- and region-specific **health-service unit costs** — cost per inpatient bed-day and per outpatient visit by facility level — estimated econometrically for essentially all WHO member states, plus generalized CEA for 479 interventions. The unit-cost engine is Stenberg/Lauer et al.'s econometric model.
- **Key limitation for your use:** These are **generic service-delivery unit costs (a bed-day, a visit), not diabetes-complication episode costs.** To get a complication cost you must multiply WHO-CHOICE unit costs by a *utilization vector* per complication (bed-days for a stroke admission, dialysis sessions/year for ESRD, outpatient visits for neuropathy, etc.). That utilization can come from the US model or from local guidelines. This "unit cost × quantity" build-up is the standard WHO-CHOICE workflow and is the most defensible way to construct LMIC complication costs from primitives.
- **Access:** Tools/models are pre-set with regional averages and are designed for a country analyst to substitute local prices/epidemiology.
- Links: [WHO-CHOICE FAQ/landing](https://www.who.int/news-room/questions-and-answers/item/who-choice-frequently-asked-questions) · [2021 update special issue (IJHPM, open access)](https://www.ijhpm.com/article_4139.html) · [Stenberg et al. 2018, econometric country-specific inpatient/outpatient unit costs, Cost Eff Resour Alloc](https://resource-allocation.biomedcentral.com/articles/10.1186/s12962-018-0095-x)

### WHO Global Health Expenditure Database (GHED)
- **What it is:** Comparable health-expenditure data for **195 countries/territories, 2000–2023** (2025 update), free and open. Includes breakdowns by financing scheme, function, provider type, and — for a **growing but incomplete subset** — **spending by disease/condition** (NCDs including diabetes).
- **Caveat:** Disease-level allocation is **voluntary and sparse** — only a minority of (mostly OECD) countries report it, coverage years vary, and large "unallocated" residuals are common. Do not count on a diabetes line for an arbitrary LMIC.
- Links: [GHED database](https://apps.who.int/nha/database/) · [WHO NCD expenditure summary brief (PDF)](https://cdn.who.int/media/docs/default-source/ncds/paper-1-nha-summary-brief.pdf) · [2025 GHED update note](https://p4h.world/en/news/who-releases-2025-update-of-the-global-health-expenditure-database/)

### Cost-of-illness literature (diabetes, by country/region)
- **IDF Diabetes Atlas (10th ed., 2021)** — diabetes-related health expenditure per person by country and region. Global mean **~$1,760/person with diabetes**; regional means: North America & Caribbean **$8,208**, Europe **$3,086**; country highs: Switzerland **$12,828**, US **$11,779**, Norway **$11,166**. Global total ~$966B. **Caveat:** these are *top-down attributed totals* (a modeled ratio of diabetic-to-non-diabetic spending), **not complication-specific**, so they anchor an overall per-patient cost but cannot populate per-complication states directly.
  - Links: [IDF Atlas — health expenditure per person by country](https://diabetesatlas.org/data-by-indicator/diabetes-related-health-expenditure/diabetes-related-health-expenditure-per-person-usd/) · [Atlas prevalence/expenditure paper (Diabetes Res Clin Pract 2021)](https://www.sciencedirect.com/science/article/pii/S0168822721004782) · [Atlas "Diabetes by region" (NCBI Bookshelf)](https://www.ncbi.nlm.nih.gov/books/NBK581937/)
- **Systematic reviews (best source for country/complication granularity):**
  - Cost-of-illness of T2DM in **low & lower-middle income countries** (BMC Health Serv Res 2018): annual direct+indirect cost/person roughly **$30–$238**; hospitalization the top direct-cost driver, then drugs; complication care costly and highly variable. [Article](https://bmchealthservres.biomedcentral.com/articles/10.1186/s12913-018-3772-8)
  - Economic burden of diabetes, **HIC vs LMIC contrast** (J Pharm Policy Pract 2024): annual cost/patient **$87–$9,581** across settings. [Article](https://www.tandfonline.com/doi/full/10.1080/20523211.2024.2322107)
  - Burden/costs of T2DM in **emerging vs established markets** (Expert Rev Pharmacoecon Outcomes Res 2020). [Article](https://www.tandfonline.com/doi/full/10.1080/14737167.2020.1782748)
  - Region-specific example: **Eastern Mediterranean Region** cost-of-illness (EMHJ 2025). [Article](https://www.emro.who.int/emhj-volume-31-2025/volume-31-issue-7/a-cost-of-illness-study-of-the-economic-burden-of-diabetes-in-the-eastern-mediterranean-region.html)
  - **Complication-specific costs** are best mined from these reviews' primary studies (many report retinopathy/nephropathy/CVD/amputation costs for specific countries), then transferred (§4).

---

## 3. Willingness-to-pay thresholds for other countries

**Do not reuse $100k/QALY abroad, and avoid the WHO 1–3× GDP rule.** Use empirical opportunity-cost thresholds where available.

### GDP-multiple thresholds (legacy, now discouraged)
- The old WHO CMH/GDP rule (cost-effective at <3× GDP/capita per DALY, "highly" at <1×) is explicitly repudiated by WHO's own authors as arbitrary and prone to endorsing unaffordable interventions. **Cite Bertram et al. 2016, "Cost–effectiveness thresholds: pitfalls and possible solutions," Bull WHO 94(12):925–930** as the reason to move away from it. Use GDP-multiples only as a coarse sensitivity bound.

### Empirical opportunity-cost thresholds (recommended)
- **Woods, Revill, Sculpher, Claxton 2016 (Value in Health)** — extrapolates the UK marginal-productivity estimate to all countries via income elasticity. Thresholds are **well below 1× GDP/capita**, rising with income (illustrative %-of-GDP ranges: low-income ~1–51%, LMIC ~4–51%, UMIC ~11–51%, higher-MIC ~33–59%). PPP-2013 USD/QALY examples: Malawi $3–116, Cambodia $44–518, El Salvador $422–1,967, Kazakhstan $4,485–8,018, UK $18,609, **US $24,283–40,112**. **All-country values are in the supplementary file (mmc1).**
  - Links: [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5193154/) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1098301516000644)
- **Ochalek, Lomas, Claxton 2018 (BMJ Global Health)** — independent derivation from cross-country mortality-vs-expenditure elasticities; central estimates **average ~0.5× GDP/capita per DALY** (often lower). Country tables in the paper's supplement and in the York CHE working paper (CHERP122).
  - Links: [PubMed](https://pubmed.ncbi.nlm.nih.gov/30483412/) · [York CHE working paper](https://ideas.repec.org/p/chy/respap/122cherp.html)
- **Pichon-Rivière et al. 2023 (Lancet Global Health)** — newest, most convenient: empirical thresholds for **174 countries** from life-expectancy and health-expenditure growth. Thresholds per QALY were **<1× GDP/capita in 168/174 countries**; per life-year they range **$78–$80,529**. Comes with a **free interactive tool (IECS)** that also lets you adjust for local health spending and GDP growth.
  - Links: [Lancet GH article](https://www.thelancet.com/article/S2214-109X(23)00162-6/fulltext) · [IECS threshold estimation tool](https://iecs.org.ar/en/thresholds/) (app at iecs.shinyapps.io/umbrales/)

### One-stop registry
- **Global Health CEA Registry — Thresholds page** aggregates published country thresholds (Woods, Ochalek, etc.): [ghcearegistry.org/orchard/thresholds](http://ghcearegistry.org/orchard/thresholds) (was returning 503 intermittently; the underlying studies above are the primary sources).

**Practical pick:** For a headline threshold use **Pichon-Rivière 2023 (IECS tool)** per country; report **Woods 2016 and Ochalek 2018** as a sensitivity range. Note the striking implication: for the US itself, the opportunity-cost threshold (~$25–40k/QALY, 2013 PPP) is far below Briody's $100k, so an honest cross-country comparison should re-run the *US* base case at an opportunity-cost threshold too, not only the new countries.

---

## 4. Currency, inflation, PPP, and cost-transfer methods

### The mechanics (currency + price year)
- **CCEMG–EPPI-Centre Cost Converter** (v1.7, Jan 2024) is the field-standard web tool. Two-stage: (1) inflate within the source country using **GDP deflators**, then (2) convert between countries using **PPP conversion rates**; both from IMF WEO / OECD. Free, browser-based.
  - Link: [eppi.ioe.ac.uk/costconversion](https://eppi.ioe.ac.uk/costconversion/)
- **Use PPP, not market exchange rates**, for cross-country health-cost comparison — market FX understates real resource use in LMICs. Consider a **health-specific price index** rather than the general GDP deflator when inflating medical costs over long spans (medical inflation ≠ general inflation), and state which you used.

### The methodology (what is/ isn't transferable)
- **ISPOR Good Research Practices — Drummond et al. 2009, "Transferability of Economic Evaluations Across Jurisdictions," Value in Health 12(4):409–418.** This is the canonical checklist. It classifies study elements by transferability and gives methods for adapting (patient-level data vs. decision-model re-parameterization).
  - Links: [ISPOR page](https://www.ispor.org/heor-resources/good-practices/article/transferability-of-economic-evaluations-across-jurisdictions) · [Value in Health full text](https://www.valueinhealthjournal.com/article/S1098-3015(10)60782-6/fulltext) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/19900249/)

### The pitfalls that matter here
1. **Unit costs and relative prices are the *least* transferable elements.** Simply PPP-converting US complication costs to an LMIC will overstate them, because health systems there substitute cheaper labor/inputs and deliver less intensive care — the *quantity and mix of care per complication differ*, not just the price. PPP fixes price levels, not practice patterns.
2. **Cost structure ≠ price.** A US dialysis year embeds US staffing ratios, capital, and drug prices; transferring the *dollar total* is wrong, transferring the *resource profile* (sessions/year, drugs, staff time) and re-pricing locally is better.
3. **Perspective drift.** US health-system costs include a large private/insurer component; many LMIC analyses use a government or societal perspective. Keep the perspective explicit and consistent with the chosen threshold.
4. **Discount rate and price year** should be localized (some jurisdictions mandate specific rates), and all costs stated in a single, named currency-year (e.g., 2024 USD PPP).

---

## 5. Practical recommendation — least-bad ways to populate complication costs

The right method differs sharply by setting. In both cases, keep the US model's *epidemiology and event rates* from GBD/the microsimulation and replace only the *cost weights*.

### A) Another high-income country (e.g., Canada, Germany, Japan, UK)
- **Best:** find a national diabetes-complication cost study or use that country's own DRG/tariff schedule to re-price each complication state (most HICs publish inpatient tariffs; NHS Reference Costs, German G-DRG, etc.). This is the ISPOR-preferred "re-parameterize with local unit costs" route.
- **Good fallback:** take the US complication cost vector (Yang 2020 / Wang 2022), keep the *resource profile*, and transfer via **CCEMG PPP conversion** to the target currency-year. Cross-country HIC transfer is where PPP conversion is most defensible because care intensity is broadly similar.
- **Threshold:** country's HTA threshold if it exists (e.g., NICE £20–30k/QALY); otherwise Pichon-Rivière/IECS value.

### B) An LMIC
- **Do not PPP-convert US dollar totals** — it will massively overstate costs and mechanically make a $60/yr intervention look implausibly cost-saving or cost-effective for the wrong reason.
- **Best available:** build complication costs bottom-up as **WHO-CHOICE local unit costs (bed-day, outpatient visit, by facility level) × a utilization vector per complication.** Draw the utilization vector from local clinical guidelines where possible, else from the US model as a transparent assumption. This keeps *prices* local and *quantities* explicit.
- **Cross-check / alternative:** anchor total per-patient diabetes cost with the **IDF Atlas** country figure and/or a **country-specific cost-of-illness study** from the systematic reviews (§2), then apportion across complication states using the *proportional* (not absolute) cost shares from the US model. Report the two approaches as a range.
- **Threshold:** **Pichon-Rivière 2023 / IECS** country value (typically <1× GDP/capita, often ~0.5×), with Woods 2016 and Ochalek 2018 as sensitivity bounds. Present results as cost per QALY/DALY against that opportunity-cost threshold, not against $100k.
- **Report honestly:** flag that LMIC complication-cost data are thin and heterogeneous, drive one-way and probabilistic sensitivity analyses off the cost inputs, and suppress/flag any complication whose local cost rests on a single small study.

### Cross-cutting advice
- State everything in **one named currency-year (e.g., 2024 USD, PPP-adjusted)** and record the deflator (prefer a health-specific index) and PPP source.
- **Re-run the US base case at an opportunity-cost threshold** so cross-country comparisons are apples-to-apples; the $100k/QALY US convention is far above the US health-system opportunity cost (~$25–40k/QALY in Woods 2016).
- The intervention cost ($60/yr vitamin D3) also needs local re-pricing — generic cholecalciferol is cheap and its price varies less than complication care, but source a local procurement/retail price rather than assuming the US figure.

---

### Key sources at a glance
- IHME DEX (US only): http://www.healthdata.org/dex
- IHME Global Health Spending / FGH: https://www.healthdata.org/data-tools-practices/interactive-visuals/financing-global-health
- WHO-CHOICE unit costs (Stenberg 2018): https://resource-allocation.biomedcentral.com/articles/10.1186/s12962-018-0095-x
- WHO GHED: https://apps.who.int/nha/database/
- IDF Diabetes Atlas expenditure: https://diabetesatlas.org/data-by-indicator/diabetes-related-health-expenditure/diabetes-related-health-expenditure-per-person-usd/
- LMIC diabetes cost-of-illness review: https://bmchealthservres.biomedcentral.com/articles/10.1186/s12913-018-3772-8
- Woods 2016 thresholds: https://pmc.ncbi.nlm.nih.gov/articles/PMC5193154/
- Ochalek 2018 thresholds: https://pubmed.ncbi.nlm.nih.gov/30483412/
- Pichon-Rivière 2023 (174 countries) + IECS tool: https://www.thelancet.com/article/S2214-109X(23)00162-6/fulltext · https://iecs.org.ar/en/thresholds/
- ISPOR transferability (Drummond 2009): https://www.valueinhealthjournal.com/article/S1098-3015(10)60782-6/fulltext
- CCEMG–EPPI cost converter: https://eppi.ioe.ac.uk/costconversion/