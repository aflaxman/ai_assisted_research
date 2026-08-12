# Research Note 2: Methodological Landscape (Quantitative Political Science)

*Deep-research report compiled by an AI research agent, 2026-08-12. Method-by-method
review of causal-inference designs applicable to estimating the effect of AIPAC
association on primary vote shares, with canonical citations, assumptions, software,
and known effect sizes from the spending/endorsement literature.*

## 0. Framing: What the Question Actually Asks, and Why It Is Hard

**The empirical setting.** AIPAC's PAC plus its super PAC (United Democracy Project,
UDP) spent [$104.3M+ in the 2026 cycle](https://readsludge.com/2026/08/03/aipac-tops-100-million-in-election-spending-for-second-straight-cycle/)
(vs. ~$61.4M in 2024, of which ~$34.8M was independent expenditure in Democratic House
races). Marquee interventions: ~$14.5M against Jamaal Bowman and ~$8.5–9M against Cori
Bush in 2024 (both lost);
[$30.6M against Abdul El-Sayed in the 2026 Michigan Senate primary](https://www.commondreams.org/news/aipac-2026-midterm-spending)
— its largest single-race outlay ever — plus funds routed through shell PACs (e.g.,
"Elect Chicago Women," $5.3M across four Illinois races, going 2–2). One tracker scores
AIPAC's contested 2026 Democratic primaries at
[3 wins, 4 losses](https://insidepoliticalmoney.com/news/aipacs-2026-scorecard-in-democratic-primaries-so-far-a-quick-rundown/).
Meanwhile, polling reported by
[Common Dreams](https://www.commondreams.org/news/democrats-could-never-support-aipac-candidate)
suggests nearly half of Democrats in competitive primary districts say they "could
never" support an AIPAC-backed candidate — i.e., in 2026 the endorsement may function as
a *negative cue* for part of the primary electorate, the core of the counterfactual
being asked. **No peer-reviewed causal estimate of AIPAC endorsement/spending effects
exists as of August 2026** — searches surface only journalism and trackers — so this
analysis would be novel.

**Three distinct treatments hide inside "AIPAC association," and each implies a
different estimand:**

1. **Receiving the AIPAC endorsement / bundled donations** (candidate-level, mostly safe
   incumbents — AIPAC endorsed 118 Democratic incumbents in 2024 but only ~5
   non-incumbents).
2. **Benefiting from UDP independent expenditure** (concentrated in a handful of
   contested primaries).
3. **Being targeted by UDP spending against you** (the Bowman/Bush/El-Sayed treatment —
   the mirror image).

The user's counterfactual — "how many more votes if candidates had *rejected* AIPAC" —
is a candidate-choice intervention. Two conceptual complications: (a) rejecting the
endorsement plausibly triggers *retaliatory* UDP spending, so the "control" condition is
not "no AIPAC" but "AIPAC on the other side" — the counterfactual must be specified as a
policy of the candidate, not a surgical deletion of one variable (a violation of
treatment-variation irrelevance / a consistency-assumption issue); and (b)
SUTVA/interference: within a race, one candidate's treatment is the opponent's exposure
(zero-sum vote shares), and across races, national narrative spillovers (e.g., Gaza
salience) affect untreated races. The design should define the unit as the
*race-candidate dyad* and the estimand as the effect on the treated candidate's share
against the same field.

**The selection problem is two-sided.** AIPAC endorses candidates likely to win anyway
(positive selection → naive comparisons overstate benefits) and targets
incumbents/candidates it believes are beatable, in races where marginal dollars matter
(strategic targeting → naive comparisons of targeted vs. untargeted incumbents overstate
the harm of being targeted). This is the classic simultaneity/endogeneity problem of the
campaign-spending literature (money flows to close races), compounded by selection on
*unobservables* (private polling). Every method below is a different way of buying
identification against this.

---

## 1. Method-by-Method Assessment

### 1.1 Selection-on-Observables: Modern Matching and Weighting

**What it identifies:** ATT of AIPAC endorsement/targeting under conditional
ignorability — treatment independent of potential vote shares given observed covariates.

**Canonical citations:**
- Hainmueller (2012), "Entropy Balancing for Causal Effects," *Political Analysis*
  20(1): 25–46 ([paper](https://web.stanford.edu/~jhain/projects/4_project_ebal/)) —
  reweights controls so covariate moments (means, variances, skew) exactly match the
  treated group, avoiding the propensity-score "balance-checking treadmill."
- Imai & Ratkovic (2014), "Covariate Balancing Propensity Score," *JRSS-B* 76(1):
  243–263 — estimates the propensity score subject to balance conditions; robust to mild
  PS misspecification.
- Supporting: Ho, Imai, King, Stuart (2007, *Political Analysis*) on preprocessing;
  Iacus, King, Porro (2012) CEM; doubly robust AIPW (Robins et al.) as the estimation
  layer.

**Mapping to AIPAC:** Build a candidate-cycle dataset of all 2018–2026 Democratic
House/Senate primaries. Covariates: incumbency and tenure, prior primary/general
margins, Bonica CFscore ideology, Squad/CPC membership, Israel-policy positions (roll
calls on Iron Dome supplemental, ceasefire statements), fundraising through Q1
(pre-treatment), small-dollar share, district partisanship (Cook PVI), district
demographics (incl. Jewish, Arab-American, Black, and college-educated population
shares), opponent quality (prior office, Porter-Treul-style coding), presence of other
super PACs (crypto PACs
[spent in lockstep with UDP in 2026](https://www.thenation.com/article/politics/crypto-ai-aipac-super-pacs-democratic-primaries-dark-money-2026/)
— a critical confounder). Entropy-balance targeted incumbents against untargeted
ideologically similar incumbents; estimate weighted regression of primary vote share on
treatment.

**Key weakness:** AIPAC's targeting uses private polling and district intelligence you
cannot observe — precisely the confounder that motivates the sensitivity analysis in
§1.6. Also, with ~5–10 heavily treated races per cycle, this is a *small-treated-N*
problem; report exact balance diagnostics and use randomization-style inference
(Fisherian permutation of treatment among the balanced pool).

**Software:** R `ebal`, `WeightIt`, `MatchIt`, `CBPS`, `cobalt` (diagnostics); Python
`DoWhy`/`causallib`.

### 1.2 Difference-in-Differences / Event Studies with Staggered Treatment

**What it identifies:** ATT for treated incumbents under (conditional) parallel trends:
absent AIPAC targeting, targeted incumbents' primary vote shares would have evolved like
comparison incumbents'.

**Canonical citations:**
- Callaway & Sant'Anna (2021), "Difference-in-Differences with Multiple Time Periods,"
  *Journal of Econometrics* 225(2): 200–230
  ([paper](https://arxiv.org/abs/1803.09015)) — group-time ATTs `ATT(g,t)` with
  never/not-yet-treated controls; outcome-regression, IPW, or doubly robust.
- Sun & Abraham (2021), "Estimating Dynamic Treatment Effects in Event Studies with
  Heterogeneous Treatment Effects," *Journal of Econometrics* 225(2) —
  interaction-weighted event-study estimator fixing contaminated TWFE leads/lags.
- de Chaisemartin & D'Haultfœuille (2020), "Two-Way Fixed Effects Estimators with
  Heterogeneous Treatment Effects," *AER* 110(9); their
  [2023 survey](https://arxiv.org/pdf/2112.04565) (*Econometrics Journal*) catalogs the
  heterogeneity-robust estimators. Also Goodman-Bacon (2021, *J. Econometrics*)
  decomposition; Borusyak, Jaravel, Spiess (2024, *ReStud*) imputation estimator; Roth,
  Sant'Anna, Bilinski, Poe (2023, *J. Econometrics*) "What's Trending in
  Difference-in-Differences" as the practitioner synthesis.
- **Parallel-trends sensitivity:** Rambachan & Roth (2023),
  ["A More Credible Approach to Parallel Trends,"](https://academic.oup.com/restud/article-abstract/90/5/2555/7039335)
  *ReStud* 90(5): 2555–2591 — bounds post-treatment PT violations by pre-trend
  magnitudes; essential companion.

**Mapping to AIPAC:** Panel of incumbent Democrats' primary vote shares by cycle
(2014–2026). Treatment onset is staggered: Bowman/Bush cohort treated 2024,
El-Sayed/Thanedar-race cohort 2026, some (Omar, Tlaib) threatened but not (yet) hit with
major IE — a genuine staggered-adoption structure where TWFE would use already-treated
units as controls (the exact pathology Goodman-Bacon diagnoses). Estimate `ATT(g,t)`
with never-treated progressive incumbents as controls, conditioning on CFscores and
district covariates (Callaway–Sant'Anna's conditional PT). Event-study leads test
whether targeted incumbents were already declining (they plausibly were — Bowman's
redistricting into Westchester predates UDP money). Caveats: few treated units per
cohort (use wild-cluster bootstrap or permutation inference); primary vote share is only
defined when there *is* a contested primary — selection into contestation is itself an
outcome (bound it, or model contest emergence as a first stage, à la Hirano & Snyder).

**Software:** R `did` (Callaway–Sant'Anna), `fixest::sunab`, `did2s`, `didimputation`,
`HonestDiD`; Stata `csdid`, `eventstudyinteract`; Python `pyfixest`, `differences`.

### 1.3 Synthetic Control / Generalized Synthetic Control

**What it identifies:** Treated-unit counterfactual trajectories under a latent factor
model — allows selection on unobserved, time-varying factor loadings (e.g.,
"vulnerability to national pro-Israel mobilization"), which DiD's parallel trends rules
out.

**Canonical citations:**
- Abadie, Diamond, Hainmueller (2010, *JASA*; 2015, *AJPS*); Abadie (2021),
  ["Using Synthetic Controls,"](https://www.aeaweb.org/articles?id=10.1257/jel.20191450)
  *JEL* 59(2): 391–425 (feasibility/data requirements).
- Xu (2017),
  ["Generalized Synthetic Control Method,"](https://yiqingxu.org/papers/english/2016_Xu_gsynth/Xu_PA_2017.pdf)
  *Political Analysis* 25(1): 57–76 — interactive fixed-effects model estimated on
  controls, counterfactuals imputed for multiple treated units with variable timing;
  built-in cross-validation ([gsynth manual](https://yiqingxu.org/packages/gsynth/); now
  a wrapper of `fect`).
- Ben-Michael, Feller, Rothstein (2021),
  ["The Augmented Synthetic Control Method,"](https://www.tandfonline.com/doi/abs/10.1080/01621459.2021.1929245)
  *JASA* 116(536): 1789–1803 — bias correction when pre-treatment fit is imperfect (it
  will be, with few pre-periods).

**Mapping to AIPAC:** For each targeted incumbent, construct a synthetic incumbent from
the donor pool of untargeted progressive incumbents, matching on 3–5 pre-cycles of
primary vote share, general share, fundraising, CFscore. Honest limitation: primaries
occur biennially and many incumbents ran unopposed in past primaries, so T₀
(pre-periods) is short and the outcome series is gappy — gsynth/fect with covariates and
`augsynth` bias correction are better suited than classic SCM; interpret as a robustness
arm, not the lead design. Inference: conformal/placebo permutation tests (Chernozhukov,
Wüthrich, Zhu 2021 *JASA*; built into `fect`).

**Software:** R `gsynth`, `fect`, `augsynth`, `Synth`/`tidysynth`; Python `pysyncon`,
`SparseSC`.

### 1.4 Regression Discontinuity: What It Can and Cannot Do Here

Close-race RD identifies effects **of barely winning** on downstream outcomes — not the
effect of endorsements on votes — so it is *not* the estimator for the headline
question. But it earns its place in three supporting roles:

1. **Downstream-consequences arm:** Hall (2015),
   ["What Happens When Extremists Win Primaries?"](https://www.andrewbenjaminhall.com/Hall_APSR.pdf)
   *APSR* 109(1): 18–42, uses primary close-race RD: nominating the extremist over the
   moderate cuts general-election vote share ~9–13 pp and win probability 35–54 pp.
   Analogue: RD on close AIPAC-candidate-vs-progressive primaries → general-election
   consequences of the AIPAC-backed nominee (does the money that wins the primary cost
   the party in November?). Also Hall & Thompson (2018, *APSR*) on turnout mechanisms.
2. **Endorsement-timing / threshold designs:** AIPAC endorsement decisions cluster
   around observable triggers (e.g., specific roll-call votes, statement deadlines); a
   discontinuity-in-time or dose-threshold design (UDP enters only above an
   internal-polling competitiveness threshold) is conceivable but the running variable
   is unobserved/manipulable — treat as exploratory. Spending-threshold RDs face the
   same objection: spending is continuously chosen by a strategic actor, violating
   no-precise-control.
3. **Validity caution for any RD use:** Caughey & Sekhon (2011, *Political Analysis*)
   show bare winners/losers in close House races differ on pre-treatment covariates;
   [de la Cuesta & Imai (2016)](https://imai.fas.harvard.edu/research/files/RD.pdf),
   *Annual Review of Political Science*, clarify that continuity — not local
   randomization — is the required assumption; Eggers et al. (2015, *AJPS*, 40,000 close
   races) largely rehabilitate electoral RD outside the U.S. House financial-advantage
   context. Primary elections, with strategic late money (exactly what UDP deploys), are
   a setting where sorting near the threshold is a live concern.

**Software:** R `rdrobust`, `rddensity` (McCrary/Cattaneo-Jansson-Ma manipulation
tests), `rdlocrand`.

### 1.5 Instrumental Variables: Candidates and Their Credibility Problems

The spending literature's IV tradition —
[Gerber (1998, *APSR*)](https://www.jstor.org/stable/2585668) instrumented Senate
spending with candidate wealth and state population; Erikson & Palfrey (1998, *JOP*;
[2000, *APSR*](https://www.cambridge.org/core/journals/american-political-science-review/article/abs/equilibria-in-campaign-spending-games-theory-and-data/CF042FCAFB51C915DD00E2F3CA9E265D))
used equilibrium restrictions and zero-covariance assumptions — shows how hard valid
instruments are here. Candidate instruments for AIPAC exposure, and their problems:

- **District Jewish population share** (as instrument for AIPAC endorsement propensity):
  fails exclusion catastrophically — Jewish population share directly affects primary
  electorates' Israel-policy preferences, media environments, and donor pools. Usable
  only as a *covariate/moderator*, never an instrument.
- **Donor-network exposure** (share of past donors connected to pro-Israel bundling
  networks via DIME): predictive of treatment, but donor networks proxy candidate
  ideology and fundraising capacity → direct outcome paths. A **shift-share/Bartik**
  construction (pre-period donor shares × national shock in pro-Israel giving
  post-October 2023) is the most defensible version; credibility then rests on
  [Borusyak, Hull, Jaravel (2022, *ReStud*)](https://arxiv.org/pdf/1806.01221)
  quasi-random shocks or Goldsmith-Pinkham, Sorkin, Swift (2020, *AER*) exogenous shares
  logic — and here shares (who your pro-Israel donors were) are plainly not exogenous to
  your Israel politics. See also the survey of
  [shift-share designs in political science](https://arxiv.org/html/2603.00135)
  (Park/Xu). Verdict: report as a bounded, heavily caveated arm at best.
- **Redistricting shocks**: plausibly the best quasi-experiment available.
  Court-ordered/commission redistricting exogenously changes a progressive incumbent's
  electorate (Bowman's 2022 map is the motivating case) and thereby AIPAC's targeting
  calculus. But redistricting changes the electorate *directly* — exclusion fails for
  vote share as outcome. Better used as (i) a source of exogenous variation in
  *targeting probability* for a marginal-treatment-effect analysis, or (ii) a
  covariate-shock to be controlled, not exploited.
- General caution: with few treated units, 2SLS is badly biased with weak instruments;
  use Anderson–Rubin confidence sets if any IV arm is run.

**Software:** R `ivreg`/`fixest`, `ShiftShareSE`, weak-IV: `ivmodel`.

### 1.6 Sensitivity Analysis for Unobserved Confounding (Mandatory Layer)

Given that AIPAC's private polling is the archetypal unobserved confounder, every
observational estimate should ship with:

- **Cinelli & Hazlett (2020)**,
  ["Making Sense of Sensitivity: Extending Omitted Variable Bias,"](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12348)
  *JRSS-B* 82(1): 39–67: report the **robustness value** (RV) — the minimum partial R² a
  confounder would need with both treatment and outcome to overturn the estimate — plus
  benchmarking against observed covariates ("a confounder 3× as strong as prior-cycle
  margin would/would not kill the result"). Contour plots via
  [`sensemakr`](https://github.com/carloscinelli/sensemakr) (R, Python, Stata, Shiny).
- **VanderWeele & Ding (2017)**,
  ["Sensitivity Analysis in Observational Research: Introducing the E-Value,"](https://www.acpjournals.org/doi/10.7326/M17-1485)
  *Annals of Internal Medicine* 167(4): 268–274 — risk-ratio-scale analogue, natural if
  the outcome is binarized to win/lose (`EValue` R package).
- **Rosenbaum (2002, *Observational Studies*) bounds** for the matched designs — Γ = the
  odds multiplier of differential treatment assignment within matched pairs needed to
  explain the result (`rbounds` R; `sensatt`/`mhbounds` Stata).
- For the DiD arm, Rambachan–Roth honest bounds (§1.2) are the parallel-trends-specific
  sensitivity analysis.

### 1.7 Bayesian Hierarchical Models for Vote Shares

**What they contribute:** Not identification (they inherit the design's assumptions) but
principled *estimation and uncertainty propagation* across heterogeneous races — partial
pooling across ~30–50 AIPAC-involved races and hundreds of controls, cycle and state
varying intercepts, and treatment-effect heterogeneity (by incumbency, district
demographics, spending dose).

**Modeling the outcome correctly:** primary vote share is a proportion (often
multi-candidate). Options: (a) **beta regression** on the treated candidate's share
(Ferrari & Cribari-Neto 2004, *Journal of Applied Statistics* 31: 799–815; hierarchical
via `brms` `Beta()` family); (b) **Dirichlet/multinomial-logit compositional models**
for full multi-candidate fields — the lineage of Katz & King (1999),
["A Statistical Model for Multiparty Electoral Data,"](https://gking.harvard.edu/publication/a-statistical-model-for-multiparty-electoral-data/)
*APSR* 93(1): 15–32, with the practical seemingly-unrelated-regression approximation of
Tomz, Tucker, Wittenberg (2002, *Political Analysis*); `DirichletReg` in R; (c) binomial
counts (votes for candidate / total votes) with overdispersion — cleanest for converting
to *votes* later. Multilevel machinery: Gelman & Hill (2007); estimation in **Stan** via
`brms`/`rstanarm`, or `PyMC`/`numpyro`. Weakly informative priors on the treatment
coefficient double as regularization for the small-treated-N problem, and the posterior
feeds §3 aggregation directly.

**MRP connection:**
[multilevel regression + poststratification](https://sites.stat.columbia.edu/gelman/research/unpublished/MRT(1).pdf)
(Gelman & Little 1997; Park, Gelman, Bafumi 2004, *Political Analysis*; Lax & Phillips
2009, *AJPS*; Wang, Rothschild, Goel, Gelman 2015 on non-representative polls) is the
bridge from *survey* estimates of the endorsement penalty (§1.8) to *district-level*
electoral quantities: model the individual-level penalty as a function of
demographics/partisanship, then poststratify onto each district's primary-electorate
composition (from voter files/ACS/CES).

### 1.8 Survey Experiments / Conjoint Designs (Complementary Experimental Arm)

**What they identify:** The *cue effect* of the AIPAC label itself — the AMCE of
"endorsed by AIPAC" (vs. no endorsement, vs. rejection of AIPAC money, vs. other
endorsers as active controls) on choice probability among primary voters — randomized,
hence unconfounded, at the cost of external validity (stated preference, forced exposure
vs. real-world awareness).

**Canonical citations:**
- Hainmueller, Hopkins, Yamamoto (2014),
  ["Causal Inference in Conjoint Analysis,"](https://www.cambridge.org/core/journals/political-analysis/article/causal-inference-in-conjoint-analysis-understanding-multidimensional-choices-via-stated-preference-experiments/414DA03BAA2ACE060FFE005F53EFF8C8)
  *Political Analysis* 22(1): 1–30 (AMCE framework).
- Hainmueller, Hangartner, Yamamoto (2015, *PNAS*) — conjoints track behavioral
  benchmarks well (the external-validity defense).
- Bansak, Hainmueller, Hopkins, Yamamoto (2018, *Political Analysis*; 2021 chapter) —
  design best practices, number of tasks/attributes.
- Leeper, Hobolt, Tilley (2020, *Political Analysis*) — subgroup comparisons (marginal
  means, not AMCE differences) for heterogeneity by, e.g., Democratic primary voters'
  Gaza attitudes.
- Abramson, Koçak, Magazinnik (2022, *AJPS*) — caution: AMCEs aggregate over preference
  intensity; report marginal means and distributional summaries.
- Precedent for endorsement conjoints/vignettes: the
  [Trump-endorsement survey-experiment literature](https://www.cambridge.org/core/journals/ps-political-science-and-politics/article/causal-effects-of-a-trump-endorsement-on-voter-preferences-in-a-general-election-scenario/2F6A370176BD854C2EA52736E98F2E1D)
  (*PS*), and cue studies in §2.

**Mapping to AIPAC:** Sample validated Democratic primary voters (match to voter file);
candidate profiles randomize AIPAC endorsement/funding acceptance alongside race,
experience, issue positions; estimate the endorsement-label penalty/premium and its
heterogeneity; then MRP the individual-level effect onto actual district primary
electorates. This triangulates the observational estimates: the observational designs
capture the *full bundle* (money → ads → salience), the conjoint isolates the *label*.
The difference between them is itself informative (an estimate of the pure-spending
channel).

**Software:** R `cjoint`, `cregg`, `factorEx`, `cbcTools`; design declaration via
`DeclareDesign` (Blair, Cooper, Coppock, Humphreys 2019, *APSR*).

---

## 2. Substantive Literature: Spending and Endorsement Effects (with Known Effect Sizes)

**Classic spending-effects debate (general elections, mostly House/Senate):**
- Jacobson (1978, *APSR*; 1990, *AJPS*
  ["The Effects of Campaign Spending in House Elections"](https://www.press.umich.edu/pdf/0472099213-ch8.pdf)):
  challenger spending buys votes, incumbent spending appears nearly impotent — the
  famous asymmetry, driven by diminishing returns to (already-known) incumbents and
  reverse causality (threatened incumbents spend more).
- Green & Krasno (1988, *AJPS*),
  ["Salvation for the Spendthrift Incumbent"](https://www.semanticscholar.org/paper/d1320df2ad9d3b0259131b000adea04dc1a00bfd):
  correcting for challenger quality and reciprocal causation, incumbent spending effects
  are substantial, at times on par with challenger spending.
- Gerber (1998, *APSR* 92(2)): IV estimates (candidate wealth, population) — incumbent
  and challenger Senate spending have roughly **equal marginal products**.
- [Erikson & Palfrey (1998 *JOP*; 2000 *APSR*)](https://www.cambridge.org/core/journals/american-political-science-review/article/abs/equilibria-in-campaign-spending-games-theory-and-data/CF042FCAFB51C915DD00E2F3CA9E265D):
  game-theoretic equilibrium identification; spending effects real for both sides,
  largest in close races (which is where the money goes — the simultaneity in a
  nutshell).
- Levitt (1994, *JPE*): repeat-matchup design → very small effects (~0.3 pp per $100k
  challenger, ~0.1 pp incumbent) — the pessimistic bound.
- Kalla & Broockman (2018),
  ["The Minimal Persuasive Effects of Campaign Contact in General Elections,"](https://www.cambridge.org/core/journals/american-political-science-review/article/abs/minimal-persuasive-effects-of-campaign-contact-in-general-elections-evidence-from-49-field-experiments/753665A313C4AB433DBF7110299B7433)
  *APSR* 112(1): best estimate ≈ **zero in general elections** across 49 field
  experiments — but with explicit carve-outs for low-information settings, unusual
  positions, and early persuasion; **primaries (no party cue) are exactly the setting
  where persuasion effects should survive**, the theoretical warrant for expecting
  nonzero AIPAC effects.
- Sides, Vavreck, Warshaw (2022, *APSR*), TV advertising effects: small but nonzero,
  larger down-ballot — a useful dose-response prior (fractions of a point per ad-share
  shift) for calibrating what $15M of primary IE could plausibly buy.
  Equilibrium/structural work on super PACs finds
  [muted equilibrium effects due to competitive response](https://eller.arizona.edu/sites/default/files/2024-09/Econ-WP-24-01.pdf).

**Endorsements as cues in primaries:**
- Dominguez (2011),
  ["Does the Party Matter? Endorsements in Congressional Primaries,"](https://www.researchgate.net/publication/258180433_Does_the_Party_Matter_Endorsements_in_Congressional_Primaries)
  *Political Research Quarterly*: party-elite endorsements strongly associated with
  primary vote share and winnowing.
- Kousser, Lucas, Masket, McGhee (2015),
  ["Kingmakers or Cheerleaders? Party Power and the Causal Effects of Endorsements,"](https://journals.sagepub.com/doi/abs/10.1177/1065912915595882)
  *PRQ* 68(3): California pre-primary party endorsements — careful causal designs find
  meaningful (roughly high-single-digit to double-digit point) boosts in low-information
  legislative primaries; effects shrink as information rises.
- Boudreau & MacKenzie (2014, *AJPS*, party cues vs. policy information) and Boudreau,
  Elmendorf, MacKenzie (2015,
  [*Election Law Journal*](https://journals.sagepub.com/doi/abs/10.1089/elj.2013.0238)):
  cues move low-information voters; information can neutralize or reverse cue effects —
  directly relevant to whether "AIPAC-backed" works as positive cue, negative cue, or
  non-cue depending on voter awareness, and to why 2026 (high Gaza salience) may differ
  from 2022.
- Hirano & Snyder (2019),
  [*Primary Elections in the United States*](https://www.cambridge.org/core/books/primary-elections-in-the-united-states/C15A9969FFE0323F1751A79B917D8B81),
  Cambridge UP: the definitive empirical treatment — primary incumbency advantage, the
  rarity of successful primarying, candidate quality selection, and (with
  newspaper-endorsement data) how endorsements historically structured primary choice.
  Baseline fact: incumbent defeats in primaries are rare events, so the Bowman/Bush
  losses are tail outcomes needing careful counterfactuals.
- Boatright (2013), [*Getting Primaried*](https://press.umich.edu/Books/G/Getting-Primaried2),
  Michigan UP: interest-group-driven primary challenges are rare and succeed only when
  groups concentrate resources on few incumbents — precisely UDP's playbook.
- Hall & Snyder (2015, "Candidate Ideology and Electoral Success," working paper) and
  Hall (2015 *APSR*, above): moderates outperform extremists electorally — matters
  because AIPAC's *stated* selection criterion correlates with moderation, entangling
  the endorsement effect with an ideology effect; Bonica (2014),
  ["Mapping the Ideological Marketplace,"](https://onlinelibrary.wiley.com/doi/abs/10.1111/ajps.12062)
  *AJPS* 58(2): 367–386, and the [DIME database](https://data.stanford.edu/dime)
  (CFscores for candidates *and* PACs, 100M+ contribution records) are the standard tool
  to control for it — and to build the donor-network exposure measures in §1.5.

---

## 3. Aggregation: From Vote-Share Effects to "Votes"

The headline number is `Σ_races [Δshare_r × TotalVotes_r]`, but three subtleties:

1. **The denominator is endogenous.** UDP spending plausibly changes *turnout*, not just
   shares (Hall & Thompson 2018 show turnout mechanisms dominate in extremist-nominee
   penalties). Estimate treatment effects on log total primary votes alongside share
   effects; report the votes-based estimand as
   `Δ(candidate votes) = share₁·T₁ − share₀·T₀` with both counterfactual objects, not
   `Δshare × T_observed`.
2. **Uncertainty propagation.** Never multiply point estimates. Use the posterior (or
   simulation-based, King, Tomz, Wittenberg 2000, *AJPS*, "Making the Most of
   Statistical Analyses") draws of race-level effects, propagate through the votes
   arithmetic per draw, and sum *within draws* — this preserves cross-race correlation
   from shared hyperparameters (a national-swing component makes race effects positively
   correlated; independent summation would understate total-vote uncertainty badly).
   Report the posterior distribution of total votes, not a point.
3. **Poststratification for the survey arm.** Conjoint AMCEs are per-exposed-voter;
   scale by (a) estimated share of the primary electorate aware of the endorsement
   (measurable in the survey), and (b) district primary-electorate composition via
   [MRP](https://bookdown.org/jl5522/MRP-case-studies/introduction-to-mrp.html) over
   turnout-adjusted poststratification cells (voter-file validated primary turnout by
   age × race × education × party). The observational and MRP-experimental aggregates
   should be presented side by side as bracketing estimates.

---

## 4. Multiverse Discipline and Pre-Registration

- **Specification curve analysis:** Simonsohn, Simmons, Nelson (2020), "Specification
  Curve Analysis," *Nature Human Behaviour* 4: 1208–1214
  ([overview](https://bookdown.org/mike/data_analysis/specification-curve-analysis.html);
  [software](http://urisohn.com/specification-curve/)): enumerate all defensible
  specifications (treatment definitions: endorsement-only / any UDP IE / IE > $1M;
  samples: incumbents-only / all candidates; outcomes: share, log votes, win;
  estimators: §1.1–1.3; covariate sets), plot the estimate distribution, and use their
  permutation-based joint inference. Steegen et al. (2016, *Perspectives on
  Psychological Science*) multiverse framing. Especially vital here because the
  politically charged question invites motivated specification choice in both
  directions. **Software:** R [`specr`](https://masurp.github.io/specr/), `multiverse`;
  Python `specification_curve`.
- **Pre-registration for observational work:** register a pre-analysis plan on
  [OSF](https://osf.io) or the [EGAP registry](https://egap.org/registry/) (which
  explicitly accepts observational designs in governance/politics) *before* the
  remaining 2026 primaries conclude — a genuine prospective element: several treated
  races (MA, NH, RI, DE primaries post-August 12) have not yet occurred, so primary
  hypotheses, treatment definitions, covariate sets, and the specification universe can
  be timestamped pre-outcome. Standards and pitfalls: Ofosu & Posner (2021),
  ["Pre-Analysis Plans: An Early Stocktaking,"](https://www.cambridge.org/core/journals/perspectives-on-politics/article/preanalysis-plans-an-early-stocktaking/94E7FAE76001C45A04E8F5E272C773CE)
  *Perspectives on Politics* (≈50% of early PAPs under-specified models — specify the
  exact estimator and SE procedure); Monogan (2015) on preregistration in political
  science; blind-to-outcome design assembly per Rubin ("design trumps analysis").

---

## 5. Recommended Design Architecture (Synthesis)

1. **Primary design:** Callaway–Sant'Anna staggered DiD on incumbents' primary vote
   shares (targeted vs. never-targeted progressive incumbents), conditional PT on
   CFscore + district covariates, wild-bootstrap/permutation inference, Rambachan–Roth
   honest bounds.
2. **Cross-sectional arm:** entropy-balanced / CBPS-weighted ATT for the full 2026
   candidate field (captures non-incumbent races DiD cannot), doubly robust estimation,
   Cinelli–Hazlett RVs benchmarked against prior-margin and crypto-PAC-presence
   covariates, Rosenbaum bounds on matched pairs.
3. **Panel robustness arm:** gsynth/fect + augmented SCM for the ~6–8 heavily targeted
   incumbents with adequate pre-periods.
4. **Experimental arm:** pre-registered conjoint + vignette experiment on validated
   Democratic primary voters; MRP the label effect to district electorates.
5. **Downstream arm (separate estimand):** Hall-style close-primary RD for
   general-election consequences of AIPAC-backed nominees, with
   Caughey–Sekhon/de la Cuesta–Imai diagnostics.
6. **Aggregation:** Bayesian hierarchical (binomial/beta) meta-model pooling race-level
   effects → posterior over total votes, turnout-endogenous denominator, within-draw
   summation.
7. **Everything wrapped** in a pre-registered specification universe with a published
   specification curve; explicit discussion of interference (zero-sum within race;
   Gaza-salience spillovers across races) and of the "rejection triggers retaliation"
   consistency problem, ideally formalized as two estimands: effect of *losing AIPAC
   support* and effect of *gaining AIPAC opposition*.
