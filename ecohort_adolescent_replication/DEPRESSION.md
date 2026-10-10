# Depression in the MNH eCohort: what is open, what needs a request, what is already published

Companion to `README.md`. Written 2026-10-10.

## TL;DR

- **The paper mis-measures depression in South Africa.** Its do-file uses `anc1depression` for South
  Africa, and the eCohort team's own code defines that variable as `m1_716c`, "anxiety or depression
  *discussed* at 1st ANC visit", a provider-screening indicator. The paper's South Africa row
  (34/173 vs 116/869) reproduces exactly from `m1_716c`. Using the PHQ-9 everywhere, South African
  adolescents have the highest depression prevalence in the study (46.8% vs 33.2% adults, p = 0.001)
  and the pooled adolescent/adult gap flips from null (18.9% vs 19.6%, p = 0.74) to significant
  (31.1% vs 26.1%, p = 0.037). The Table 5 finding that "antenatal depression" predicts obstetric
  complications among adolescents (aOR 2.43) mixes the two definitions across countries.
- **Open now:** baseline PHQ-9 for all 4,068 women in four countries, plus for Ethiopia the complete
  perinatal depression series (PHQ-9 monthly in pregnancy, PHQ-2 at 2 to 4 and 6 to 8 weeks postpartum,
  PHQ-9 at 10 to 12 weeks) with IPV, delivery experience and birth outcomes, for 816 women.
- **Needs a data request (KEMRI-Wellcome DGC):** anything after the first ANC visit for Kenya and
  South Africa, and the India follow-up. Note the monthly follow-up carried the full PHQ-9 only in
  Ethiopia; Kenya, India and South Africa asked the PHQ-2 (items a and b) monthly.
- **Already published:** baseline prevalence and the screening gap (Arsenault 2024, PLOS Med), the
  Ethiopia antenatal trajectory and determinants (Wondim 2026, BMJ Open), and the adolescent/adult
  comparison with the flawed South Africa measure (Amboko 2026). Not yet published: any four-country
  postpartum depression result, any adolescent-specific depression analysis, and the Ethiopia
  pregnancy-to-postpartum trajectory (two Wondim datasets are deposited ahead of their papers).

## How depression was measured in the eCohort

From the protocol paper (Arsenault et al. 2024, Glob Health Action 17:2392352, PMC11338195) and the
team's cleaning code (github.com/QuESTHarvard/ecohorts):

| Module | Timing | Depression items |
|---|---|---|
| M1 | first ANC visit, in person | PHQ-9 (`phq9a`..`phq9i` / `m1_phq9*`); `m1_716c` provider discussed anxiety/depression; `m1_724d` referred to mental health provider; `m1_202d` prior mental health diagnosis; `anc1mental_health_drug` |
| M2 | monthly phone calls in pregnancy | Ethiopia: PHQ-9 (`m2_205a`..`m2_205i`); Kenya, India, South Africa: PHQ-2 only (`m2_205a`, `m2_205b`); `m2_508a` counselling session since last call |
| M3 | 2 to 4 weeks after birth, phone | PHQ-2 (`m3_801a`, `m3_801b`); `m3_802a` counselling since last call |
| M4 | 6 to 8 weeks, phone | PHQ-2 (`m4_302a`, `m4_302b`); `m4_703c` provider discussed anxiety/depression; `m4_704a` counselling; `m4_801i` medicine for mental health |
| M5 | 10 to 12 weeks, in person | PHQ-9 (`m5_depression_a`..`i`); EQ-5D anxiety/depression item (`m5_health_e`); `m5_803c`, `m5_804a`, `m5_901i` |

Cut-offs used by the team: PHQ-9 >= 5 "mild or worse" (`depress`, `phq9_cat`), PHQ-2 >= 3.

## What is openly available

| Source | License | Countries | Women | Depression content |
|---|---|---|---|---|
| `eCOANC1.dta`, Dataverse 10.7910/DVN/SSVXKY | CC0 | ET, IN, KE, ZA | 4,068 | baseline PHQ-9 binary (`depress`), `m1_716c` (= `anc1depression`), `m1_721`, `m1_724d`, `m1_202d`, mental health drug. No PHQ-9 items or score. No IPV. |
| `econut_analyticdata.dta`, Dataverse 10.7910/DVN/JGAXNG | CC0 | ET | 811 | baseline `depress`; PHQ-9 items at M2 rounds 1 to 8 |
| Wondim, Zenodo 10.5281/zenodo.19205479 | CC-BY | ET | 933 (long, 7 timepoints) | baseline PHQ-9 items, M2 PHQ-9 rounds 1 to 7, IPV, counselling, HRQoL, plus Stata code |
| Wondim, Zenodo 10.5281/zenodo.20730905 | CC-BY | ET | 816 (wide) | **near-complete Ethiopia M1 to M5 extract, 4,326 variables**: all of the above plus M3 PHQ-2, M4 PHQ-2, M5 PHQ-9, M3 mistreatment items, delivery, complications, birth outcomes |
| Wondim, Zenodo 10.5281/zenodo.19119541 | CC-BY | ET | 766 (long) | perinatal PHQ series through M5 plus infant HRQoL |
| `ecohort_perinatal_outcome*.dta`, Dataverse 10.7910/DVN/Q0YKOT | CC0 | 4 | 3,600 | none (mental health disorder folded into a risk-factor count) |

The Zenodo ABO file means the whole Ethiopia column of the Amboko paper, Tables 3 to 5 included,
is replicable from open data (67 adolescents with delivery data). It is also the only open source
for postpartum depression in this cohort.

## What needs a data request

Modules 2 to 5 for Kenya and South Africa, and Modules 2 to 5 for India. Request route: Data
Governance Committee, KEMRI-Wellcome Trust Research Programme (dgc@kemri-wellcome.org), per the
Dataverse terms on 10.7910/DVN/AUQ8C8 and https://kemri-wellcome.org/data. The restricted
Amboko file carries `ANCdepression` and `depression_pp` (M3 PHQ-2 >= 3) for the three African countries.

## What the open data shows (see `depression_open_data.py`, output in `depression_open_data_output.txt`)

Baseline PHQ-9 >= 5, first ANC visit:

| | Ethiopia | India | Kenya | South Africa |
|---|---|---|---|---|
| all women | 24.7% | 9.0% | 19.6% | 35.4% |
| adolescents vs adults | 18.6 vs 25.3 | 9.0 vs 9.0 | 17.5 vs 19.8 | **46.8 vs 33.2** |
| provider discussed anxiety/depression | 3.7% | 14.2% | 26.0% | 14.5% |
| referred to mental health provider | 0.1% | 10.8% | 0.4% | 0.4% |

Among screen-positive women, the share whose provider raised anxiety or depression was 3.6% in
Ethiopia, 5.6% in India, 18.5% in Kenya and 15.5% in South Africa, and was no higher than among
screen-negative women. Referral among screen-positive women was 0.4% to 2.2%.

Pooled correlates of a positive screen (crude): unmarried 34% vs 18%; unintended pregnancy 31% vs
16%; any danger sign 41% vs 15%; known HIV 41% vs 21%; prior mental health diagnosis 47% vs 22%;
self-rated health fair/poor 44% vs 20%. Wealth tertile, health literacy, parity and trimester
showed little gradient. Adolescence pooled: 27% vs 22%.

Ethiopia longitudinal (econut file, 811 women): PHQ-9 >= 5 at 37%, 32%, 28%, 29%, 31% across
monthly rounds 1 to 5 (mean gestational age 22 to 39 weeks); PHQ-9 >= 10 at 13% falling to 8%;
item 9 (thoughts of being better off dead) endorsed by 5% to 10% per round. Women screen-positive
at baseline averaged PHQ-9 5.6 during follow-up versus 3.2, and 76% versus 47% screened positive at
least once. Among women with three or more scored rounds, 55% were positive at least once but only
12% at every round, so symptoms fluctuate more than they persist.

## What is already published from this cohort on depression

1. **Arsenault et al. 2024, PLOS Medicine** (doi:10.1371/journal.pmed.1004446). Baseline, four
   countries. 22% (905/4,068) PHQ-9 positive; only 4% (Ethiopia) to 18% (Kenya) of positive women
   were screened at the visit; under 3% referred or prescribed. Depression is one of seven conditions
   in the risk-detection sub-analysis. Data: SSVXKY.
2. **Wondim et al. 2026, BMJ Open** (doi:10.1136/bmjopen-2026-120181). Ethiopia, 933 women, seven
   timepoints. PHQ-9 >= 5 at 24% baseline then 29% to 37% monthly; 59% positive at least once, 23%
   three or more times. Three GBTM classes: minimal 38%, mild fluctuating 50%, mild-to-moderate rising
   12%. GEE determinants: IPV aOR 2.43, danger signs 2.60, complications 1.87, unintended pregnancy
   1.29, first trimester 1.45, married 0.53. Only 5.6% received any counselling or support. Age was
   not modelled. Data: Zenodo 19205479.
3. **Amboko et al. 2026, PLOS GPH** (this paper). Depression appears only as a covariate: Table 1
   prevalence (flawed for South Africa, see above), Table 3 "peripartum depression" PHQ-2 > 3 at
   5.4% adolescents vs 7.1% adults, and Table 5 aOR 2.43 for obstetric complications among
   adolescents.
4. **Deposited but not yet published** (Zenodo 20730905, 19119541): Wondim's antenatal-depression
   trajectories and adverse birth outcomes, and perinatal depression and infant quality of life.
5. **Arsenault et al., "Longitudinal ANC quality and postpartum health"**: the 2025 medRxiv version is
   withdrawn; a Bulletin of the WHO 2026 version is referenced by its Zenodo supplement
   (10.5281/zenodo.23065187). Whether postpartum depression enters its "poor postpartum health"
   outcome is not visible from the abstract.

## Open questions this cohort could answer, and from which data

| Question | Open data enough? |
|---|---|
| Adolescent vs adult PHQ-9 at baseline, four countries, corrected measure | yes (eCOANC1) |
| Does adolescence predict depression after adjustment for marriage, pregnancy intention, IPV? | partly: IPV is open only for Ethiopia; four-country needs request |
| Depression and later mistreatment, complications, continuity of care | Ethiopia yes (Zenodo ABO); KE/ZA need request |
| Antenatal to postpartum trajectory, PHQ-9 at 10 to 12 weeks | Ethiopia yes (Zenodo ABO / perinatal); others need request |
| Screening, referral and treatment cascade over the perinatal period | Ethiopia yes; others need request |
| Is monthly PHQ-2 (KE/IN/ZA) a useful surrogate for PHQ-9? | Ethiopia only, both scales are present, open |
