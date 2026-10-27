# Hepatitis B birth dose, CTC and CPAD devices — a camdl replication

Replicates Seaman, Morgan, Howell, Xiao, Spearman, Sonderup, Lesi, Andersson,
Hellard & Scott (2020), "Use of controlled temperature chain and compact prefilled
auto-disable devices to reach 2030 hepatitis B birth dose vaccination targets in
LMICs: a modelling and cost-optimisation study", *Lancet Global Health* 8:e931–41,
[doi:10.1016/S2214-109X(20)30231-X](https://doi.org/10.1016/S2214-109X(20)30231-X),
using [camdl](https://github.com/vsbuffalo/camdl).

![replication summary](results/hbv_replication.png)

## Headline

Every policy conclusion of the paper replicates. The absolute DALY numbers do not:
the authors' code counts years lived with disability about 30 times over, and the
paper's "389 DALYs per 1000 births" is almost entirely that overcount.

| Claim (All GBD regions unless stated) | Paper | camdl | Verdict |
|---|---|---|---|
| Baseline birth-dose coverage | 44% | 44.2% | **Replicates** |
| Maximum coverage under current protocols (scenario 1) | 65% (IQR 64–66) | 65.0% | **Replicates** |
| Coverage with CTC and CPAD everywhere (scenario 4) | 91% | 91.3% | **Replicates** |
| 90% reachable with CTC alone only in SE Asia, E Asia & Oceania | yes | yes (90.3%; next best 89.4%) | **Replicates** |
| Cheapest route to 90% in all six regions, per year | $687.5 million | $687.5 million | **Replicates exactly** |
| … of which on CTC and CPAD | $516.5 million (75%) | $516.5 million (75%) | **Replicates exactly** |
| Increase over current expenditure | $494.0 million | $494.0 million | **Replicates exactly** |
| 90% optimum cost saving in 5 of 6 regions; South Asia ICER up to $37.49 | yes | yes; South Asia $36 per DALY (median $28.80, IQR 20.43–41.01) | **Replicates** |
| Scenario 4 cost saving in 4 regions; exceptions Central/Eastern Europe & Central Asia (ICER ≤ $4.79) and South Asia (≤ $94.58) | yes | yes; $2.5 and $91 per DALY | **Replicates** |
| 90% coverage cost saving in 40 of 46 LMICs | 40 | 38 (Peru and Colombia cost an extra $0.04–0.05 per birth; see finding 2) | **Replicates in substance** |
| CPAD not needed in Suriname and Costa Rica; CPAD share elsewhere 0.04% (Lebanon) to 84.7% (Myanmar) | yes | yes; 0.04% to 84.7% | **Replicates exactly** |
| Annual 90% cost highest in South Asia: Bhutan $10,486 to Timor-Leste $35,220 per 1000 births; Nigeria $6,501 | yes | $10,486; $35,220; $6,501 | **Replicates exactly** |
| Baseline vaccine spend per 1000 births | $1,466 (IQR 1,383–1,566) | $1,461 (IQR 1,385–1,553) | **Replicates** |
| Baseline disease-management cost per 1000 births | $11,681 (IQR 10,190–13,388) | $10,900 (IQR 9,278–12,633) | **Replicates** |
| Baseline DALYs per 1000 births | 389 (IQR 319–468) | 369 (IQR 303–435) with the authors' YLD factor; **58 (IQR 48–69)** without it | **Does not replicate** as a quantity |
| Scenario 4 DALYs averted; net saving per 1000 births | 280 (227–330); $1,924 (628–3,096) | 265 (219–314); $1,408 (90–2,590) | **Replicates** |
| 90% optimum, six regions, DALYs averted per cohort | 36.3 million (IQR 30.1–43.4) | 39.8 million (IQR 33.8–46.3) summing regions, 34.1 million (28.1–40.4) from the aggregate row; 6.3 million without the factor | **Replicates** given the factor |
| Sub-Saharan Africa share of averted burden | 80% | 70% (28.0 million, IQR 22.8–33.1, of 39.8 million) | **Replicates** |

## What was done

The paper's model is a deterministic birth cohort followed for 100 years
(Figure 1): newborns are split, by maternal HBsAg and HBeAg status, birth location
and the protection an effective birth dose gives, into immune, susceptible and
perinatally infected; infected infants progress through acute infection to chronic
infection, cirrhosis and hepatocellular carcinoma, with hepatitis-B and age-specific
all-cause mortality. Around it sits an arithmetic layer: how coverage of five
delivery strategies (cold chain or CTC in facilities; cold chain, CTC or CPAD in
the community) converts into protected births; vaccine prices and logistic
cost-coverage curves; and an SLSQP optimisation of the allocation across
strategies. The authors published their code and inputs
([GitHub](https://github.com/ChrisSeaman-Burnet/Seaman-et-al.-HBV-Birth-Dose-Model)).

- `hbv_birth_dose.camdl` is the cohort model. The age-dependent chronicity
  probability (0.885 under six months, then `exp(-0.645 age^0.455)`) and the
  22-band all-cause mortality are written as runtime expressions of `t`; the
  Figure 1 split is the `init` block; discounted YLL, YLD and disease cost are
  `quantities`. It runs on the ODE backend with fixed-step RK4 at 1/32 year.
- `hbv_model.py` ports the arithmetic layer function by function from the authors'
  `model.py`, `model_setup.py`, `costcoverage.py` and `optimisation_model.py`, and
  drives `camdl simulate`.
- `run_published_code.py` runs the authors' code unmodified on the same spreadsheet,
  so that camdl can be compared with what their code computes, separately from what
  the paper prints.
- `run_replication.py` (six regions and their aggregate: baseline, four scenarios,
  two optimisations), `run_uncertainty.py` (1000 probabilistic draws, as in the
  paper) and `run_countries.py` (the 46 LMICs below 90%) produce everything in
  `results/`.

The arithmetic layer reproduces the authors' code to the cent: the same coverage,
the same vaccine expenditure for every strategy set in every region, and the same
optimised allocations. The disease outcomes agree to about 5%, for the reason in
finding 2.

## Findings

### 1. The YLD "normalising factor" multiplies disability by about 30

In the authors' `hbv_model`, years lived with disability accumulate each time step
as

```python
exp(-discount*t) * (DALY_A*A + DALY_C*C + ...) * (2997/len(t_steps))   # normalizing factor
```

The model runs 500 steps of 0.2 years, so the factor is 5.994 where the integral of
disability-weighted prevalence needs the step length 0.2. YLD is therefore 29.97
times too large. YLL, disease costs and everything else use `dt` correctly.

camdl's `yld` quantity is the integral itself. At baseline for all GBD regions it is
9.2 per 1000 births; the authors' code gives 291.3, which is 29.97 × 9.7. Reported
DALYs follow: with the factor camdl gives 325 per 1000 births (paper 389, IQR
319–468, from the probabilistic analysis); without it, 57. Of the paper's 389 DALYs,
roughly 340 are the overcount. All results here are reported both ways
(`dalys_as_published` and `dalys`).

This changes no policy conclusion. The cost-saving verdicts depend on disease costs,
which are computed correctly; the ICERs where the interventions are not cost saving
would be about six times higher without the factor (South Asia's 90% optimum,
$36 per DALY averted, becomes about $215, still below common thresholds); and the
DALY totals scale together, so the ranking of strategies and regions is unchanged.

### 2. Forward Euler at 0.2 years overstates chronicity by 5 percentage points

The authors integrate with forward Euler at dt = 0.2 years. Acute infections leave
that state at 4 per year, and the probability of chronicity falls from 0.885 to
about 0.6 at six months of age. The Euler scheme moves the whole latent class to
acute in one step and settles 80% of acute cases per step, so most chronicity
decisions fall at ages 0.2 and 0.4 years, before the curve drops. Integrated
continuously (camdl, RK4 at 1/32 year), more of them fall after six months.

| | Authors' code | camdl |
|---|---|---|
| Chronic infections per perinatal infection | 86.6% | 81.7% |
| HBV deaths, baseline, per 1000 births | 4.05 | 3.87 |
| Disease cost, baseline, per 1000 births | $10,533 | $10,015 |

This 5% carries through every disease outcome and explains why camdl's point
estimates sit a little below the authors' medians. It also bears on the two most
marginal countries. The paper counts Peru and Colombia among the 40 cost-saving
LMICs; here reaching 90% costs them an extra $47 and $39 per 1000 births, about
4 cents per birth. Scaling camdl's disease savings up by the 5% would narrow that
to $34 and $31 but not close it, so the remaining difference is unexplained; the
paper's country ICERs carry interquartile ranges, so they may come from its
uncertainty analysis rather than a point run.

### 3. The disease model is linear, so the optimisation never needs it

There is no transmission inside the cohort, so every outcome is proportional to the
number of perinatal infections, which is closed-form in the coverage inputs. The
optimisations (maximise births reached for a budget; minimise spend for 90%
coverage) depend only on the cost-coverage curves. camdl is used as the authors
use their disease model: to value each coverage level in deaths, DALYs and costs.
The uncertainty analysis exploits the linearity, running one cohort per draw and
region and scaling by infections; `run_uncertainty.py --check` confirms the shortcut
against direct runs to 1 part in 10,000.

### 4. Small things

- The authors' spreadsheet gives the latent period as 2 weeks (`r_LA` = 24/year),
  the paper says 3 weeks, and the code overrides it to 1/dt = 5/year so that the
  Euler step empties the latent class exactly. camdl uses the spreadsheet value; the
  choice does not matter because background mortality during latency is negligible.
- The code's age-band lookup for all-cause mortality (`ceil(years)/5`) starts each
  five-year band one year early (the 10–14 rate applies from age 9.2). camdl uses
  the nominal bands. The effect on outcomes is below 1%.
- Years of life lost are counted to the end of the 100-year horizon, not to a life
  table; camdl keeps the authors' definition.
- Paper Figure 2 draws coverage on a second axis against expenditure; the figure
  above writes coverage over each bar instead.

## Uncertainty analysis

`run_uncertainty.py` draws every global parameter uniformly between its Table 1
bounds, every regional parameter uniformly between the authors' lower and upper
sheets, and HBsAg prevalence from a truncated normal, as `uncertainity.py` does.
Medians and IQRs for all GBD regions, per 1000 births, over 1000 draws:

| Strategy set | Coverage | Vaccine spend | Disease cost | DALYs, authors' YLD factor | DALYs, YLD at face value | DALYs averted (factor) | Net cost vs baseline |
|---|---|---|---|---|---|---|---|
| Baseline | 44.1% | 1,461 (1,385–1,553) | 10,900 (9,278–12,633) | 369 (303–435) | 58 (48–69) | — | — |
| Optimised baseline budget | 70.8% | 1,471 | 6,197 (5,271–7,231) | 210 (172–248) | 33 (27–39) | 159 (132–188) | −4,692 (−5,462 to −4,003) |
| Scenario 1 | 65.0% | 1,764 (1,701–1,842) | 7,441 (6,376–8,719) | 252 (207–299) | 40 (33–47) | 115 (95–138) | −3,124 (−3,682 to −2,589) |
| Scenario 2 | 72.4% | 1,916 (1,855–1,993) | 6,214 (5,331–7,324) | 211 (172–250) | 33 (28–40) | 156 (129–188) | −4,194 (−4,911 to −3,474) |
| Scenario 3 | 76.6% | 2,597 (2,489–2,707) | 5,563 (4,761–6,572) | 189 (154–224) | 30 (25–35) | 179 (148–214) | −4,189 (−5,001 to −3,356) |
| Scenario 4 | 91.3% | 7,973 (7,485–8,394) | 3,053 (2,588–3,579) | 103 (85–122) | 16 (14–20) | 265 (219–314) | −1,408 (−2,590 to −90) |
| Optimised 90% | 90.0% | 5,225 | 3,228 (2,742–3,767) | 109 (90–130) | 17 (14–21) | 259 (214–307) | −3,891 (−5,127 to −2,778) |

The paper's corresponding medians are $1,466 vaccine spend, $11,681 disease cost
and 389 DALYs at baseline, and 280 DALYs averted with $1,924 saved under scenario 4.
Vaccine spend matches; the disease outcomes sit 5–10% lower here, the Euler
difference of finding 2 compounded by the authors' optimised strategies being
simulated with its coarser scheme.

Six-region totals for the cheapest route to 90% coverage, per birth cohort:

| | Median (IQR) |
|---|---|
| DALYs averted, authors' YLD factor, summed over the six regions | 39.8 million (33.8–46.3) |
| DALYs averted, authors' YLD factor, from the aggregate "All GBD regions" row | 34.1 million (28.1–40.4) |
| DALYs averted, YLD at face value, summed over regions | 6.3 million (5.4–7.2) |
| Of which Sub-Saharan Africa (factor) | 28.0 million (22.8–33.1) |
| Net lifetime cost, summed over regions | −$324 million (−417 to −238) |

The paper reports 36.3 million (IQR 30.1–43.4) DALYs averted, 29.0 million (22.6–34.0)
of them in sub-Saharan Africa, and does not say whether the total sums the regional
rows or scales the aggregate row; it lies between the two. Sub-Saharan Africa
agrees either way.

## Since 2020: what has changed, and open questions

Notes from a literature check in October 2026, kept here for a possible
follow-up.

### What has changed

- **Still no CTC-licensed hepatitis B vaccine.** WHO's October 2023 CTC FAQ
  lists four vaccines labelled for controlled-temperature-chain use, none of
  them hepatitis B, though the birth dose has been a CTC priority since 2017.
  Knudson et al. (*Vaccines* 2026) confirm no product and no national-scale
  out-of-cold-chain use; pilot gains of 27–40 points still stand. PATH's 2026
  Ethiopia brief: 95% of stakeholders want a CTC product and would pay a little
  more. The paper's scenarios 2–4 remain hypothetical.
- **Gavi now funds the birth dose, but not the delivery the optimum needs.**
  Support launched June 2024 for 34 countries; 12 had applied by September 2025;
  Ethiopia launched nationally in November 2025. Vaccine is $0.24 a dose ($0.34
  landed) against a $0.20 co-financing floor. Gavi 6.0 protects procurement but
  cuts country cash grants 20–40%, and the paper's 90% optimum spends 75% of its
  money on community delivery. Gavi's 2018 investment case declined to fund
  CPADs on price.
- **Prices in the paper are still about right.** UNICEF (May 2025): single-dose
  vials $0.48–0.60 (paper $0.55); the 2022 microarray-patch paper quotes Uniject
  at $1.65 (paper $1.63).
- **Coverage has stalled.** Timely birth dose 43% globally in 2025 (paper's 2017
  baseline 44%), 119 countries introduced; Africa 17–18% (paper used 10% for
  sub-Saharan Africa).
- **Lower prevalence in Africa.** Riches et al. (*Lancet Global Health* 2025):
  HBsAg prevalence in antenatal women in the WHO African region 6.2% (5.3–7.2)
  for 2014–23, against the paper's 7.8%. The cohort model is linear in
  infections, so sub-Saharan benefits scale by about 0.8; the 90% optimum there
  still saves roughly $9,400 per 1,000 births. The same paper estimates 172,000
  vertical transmissions in Africa in 2022; 90% birth dose cuts them 44%
  (CI 12–78), adding maternal antiviral prophylaxis cuts them 86%.
- **Maternal antivirals change the comparator.** WHO's 2024 guidelines expanded
  tenofovir eligibility in pregnancy; Negasa et al. (BMC Public Health 2025)
  find tenofovir for all HBsAg-positive women in Ethiopia cost-effective at
  $220 per DALY. The paper excluded antivirals.
- **Dynamic models find bigger, more fragile benefits.** De Villiers, Nayagam &
  Hallett (*Nature Communications* 2021): 90% timely coverage by 2030 averts
  710,000 deaths in the 2020–30 cohorts and 41 million chronic infections to
  2100 (the cohort model here ignores onward transmission), but adult treatment
  at 40% or 80% of eligible cuts the birth dose's deaths averted by 42% or 77%.
- **Same group, same model, new device.** Seaman et al. (PLOS Global Public
  Health 2022) apply this model to microarray patches: cost-effective in nearly
  all 80 LMICs up to $5 a dose, $88.65 per DALY at that price. Seaman et al.
  (*Lancet Gastroenterology & Hepatology* 2023) is the global investment case.

### The YLD error in the 2022 code

The microarray-patch code (ChrisSeaman-Burnet/HepB-BD-MAPs-Model,
`code/model_run.py`) does **not** repeat finding 1. YLD accumulates as
prevalence × disability weight × step length × discount, and YLL is deaths in
each step × remaining life expectancy from a life table rather than the integral
of cumulative deaths to a 100-year horizon. The fix is not mentioned in the
paper. Two consequences: the 2020 and 2022 ICERs are on different scales (the
2020 "up to $37.49 per DALY" in South Asia is about $215 on the 2022 scale), and
the per-dose DALYs differ 12–30-fold between the papers (0.6 versus 0.02–0.05),
most of which is the correction rather than epidemiology. The 2022 grid has a
small inconsistency of its own: 24 sub-steps in year one, then annual steps from
year two, so the step from year 1 to year 2 uses a step length of 1/24 while the
clock jumps a year, and discounting after year one lags the clock by a year. Not
run, only read.

### Candidate questions for a follow-up model

Ordered by how much a simulation adds over arithmetic. Marked ones are the ones
I would pick up first.

1. **Birth dose versus maternal antiviral prophylaxis for a fixed budget** (★).
   Both act on HBeAg-positive, high-viral-load mothers, so each lowers the
   other's marginal value, and the paper excluded antivirals. Antenatal
   attendance is high where facility delivery and testing are low (Nigeria DHS
   cascade, "contact without protection"), so the two reach partly different
   women. Model the PMTCT cascade with correlated access; prophylaxis enters the
   camdl model as a per-mother transmission multiplier. Bears on the 2026 Gavi
   application window.
2. **Adult treatment scale-up erodes the birth-dose payoff** (★). Add a
   treatment cascade for the infected cohort under 2024 expanded eligibility and
   plausible 2030–50 coverage; see how far the cost-saving verdicts survive on
   the corrected DALY scale, where margins are thinner. Natural sensitivity
   layer on question 1.
3. **Horizontal childhood transmission and hepatitis D.** The cohort model
   credits only the perinatal infection prevented in the vaccinated child. A
   dynamic transmission model (camdl's strength) would show how much the paper
   understated benefit in sub-Saharan Africa, which carries 70–80% of it, and
   add the hepatitis D host removed with each carrier.
4. **The Gavi 6.0 funding shape.** Procurement protected, cash cut: what
   coverage does facility-only introduction reach, and what does delaying
   community strategies to 2028 or 2030 cost in deaths? Existing model plus a
   time dimension on introduction; de Villiers' 580,000-death delay estimate is
   the global benchmark.
5. **Vial presentation, wastage and timeliness at small facilities** (★). Births
   arrive at random, a ten-dose vial opened for one child is used or discarded,
   and effectiveness falls from 95% on day 1 to 52% by days 8–41. A
   discrete-event simulation of facility births, vial-opening policy, stock-outs
   and the race between birth notification and a community worker's visit gives
   wastage, timeliness and coverage as outputs rather than inputs, and shows
   where CTC adds the most value (PATH's open question). The 2020 model's
   structure gives no help here; the most novel of the eight.
6. **Facility births are rising on their own.** The paper froze the facility
   share at 2017 levels (58% in sub-Saharan Africa). Project the trend and find
   the facility share at which community strategies stop being cost saving.
   Cheap, and could overturn the regional ranking.
7. **Backtest the 65% ceiling.** Eight years of country-level WUENIC data now
   exist to test whether countries that scaled facility delivery plateaued
   where the model said. Not a new model; the cheapest credibility check.
8. **Model-structure sensitivity.** Cohort versus dynamic, horizon versus
   life-table YLL, Euler versus continuous chronicity: a deliberate comparison
   would say which numbers across the 2020, 2022 and 2023 papers are comparable.

Sources for this section: WHO CTC FAQ (2023); Knudson et al., *Vaccines* 14(7):554
(2026); PATH, "HepB birth-dose brief, Ethiopia" (2026); World Hepatitis Alliance,
Gavi birth-dose programme pages; MSF information note on birth-dose introduction
in Gavi 6.0 (April 2026); UNICEF hepatitis B vaccine price data (7 May 2025); WHO
immunization coverage fact sheet (2025); Riches et al., *Lancet Glob Health*
13:e447–58 (2025); WHO guidelines for chronic hepatitis B (2024); Negasa et al.,
*BMC Public Health* 25:1300 (2025); de Villiers, Nayagam & Hallett, *Nat Commun*
12:6223 (2021); Seaman et al., *PLOS Glob Public Health* 2(5):e0000394 (2022);
Seaman et al., *Lancet Gastroenterol Hepatol* 8:932–42 (2023).

## Files

| File | What it is |
|---|---|
| `hbv_birth_dose.camdl` | the cohort model |
| `hbv_model.py` | inputs, Figure 1 arithmetic, costs, cost-coverage curves, optimisation, camdl driver |
| `run_replication.py` | deterministic replication for the GBD regions |
| `run_uncertainty.py` | the 1000-draw probabilistic analysis |
| `run_countries.py` | the 46 LMICs below 90% coverage |
| `run_published_code.py` | runs the authors' code on the same inputs (pass the path to their `code/` directory) |
| `plot_replication.py` | the figure above |
| `data/hbv_model_inputs_revision.xlsx` | the authors' inputs, unmodified |
| `data/published_code_reference.tsv` | what the authors' code computes |
| `results/scenarios_per_1000.tsv` | every strategy set, every region, per 1000 births |
| `results/population_totals.tsv` | the same scaled to 2017 births |
| `results/comparison_published_code.tsv` | camdl against the authors' code |
| `results/uncertainty_summary.tsv` | medians and IQRs over the draws |
| `results/countries_90pct.tsv` | the country analysis |

## Running it

```bash
uv sync
uv run python run_replication.py
uv run python run_uncertainty.py 1000
uv run python run_countries.py
uv run python plot_replication.py
```

camdl 0.1.0+eb20565 (October 2026); see the camdl section of the repository
`CLAUDE.md` for building `camdl` and `camdlc`.
