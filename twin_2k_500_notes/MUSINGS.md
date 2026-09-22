# Musings: digital twins for designing disability-weight and verbal/social autopsy surveys

Companion to [README.md](README.md), which records what Twin-2K-500 and its
mega-study actually found. The question here is what the *approach* (a deep,
re-contactable human panel; LLM twins built from each person's answer history;
a standardized human-vs-twin evaluation harness) could do for two IHME survey
traditions: the GBD disability-weight (DW) surveys, and verbal autopsy (VA)
with its social-autopsy (SA) extensions.

## The stance in one paragraph

The mega-study's authors describe their twins as "hyper-rational,
quasi-omniscient versions of humans, with implicit values partly imbued by their
base LLM," and warn against deployment. I take that at face value. Twins at
r ≈ 0.2 cannot stand in for respondents, and for DW they *must not*: a
disability weight is a social judgment whose legitimacy comes from being made by
people. But almost everything expensive about developing a survey happens
before a single estimate is produced: writing and revising items, checking
comprehension, choosing which comparisons to ask, testing skip logic, deciding
what to cut. That design work needs a respondent who is roughly right, cheap,
tireless, and available at 2 a.m. That is exactly what a funhouse-mirror twin is.
Use twins to design instruments; use humans to produce estimates; and use the
twin-human gap itself as data.

## Two very different targets

**Disability weights** ask the general public to compare hypothetical people in
lay-described health states ("which person is healthier?") plus a few population
health equivalence questions to anchor the probit scale between 0 and 1. GBD 2010
used 220 states, 13,902 household respondents in Bangladesh, Indonesia, Peru,
Tanzania and the USA, and 16,328 web respondents; results correlated r ≥ 0.9
across surveys except Bangladesh (0.75). GBD 2013 added 30,660 European web
respondents and re-estimated 183 states with r = 0.992 agreement on unchanged
states. Two things follow. The aggregate ordering is robust and easy, so twins
add little there. And the lever that *does* move weights is wording: revised
lay descriptions moved complete hearing loss from 0.033 to 0.215 and neck-level
spinal cord injury from 0.369 to 0.589. DW development is, operationally, a
text-comprehension problem.

**Verbal autopsy** asks a proxy respondent to recall a decedent's signs,
symptoms, and care history, then assigns a cause by physician review or
algorithm (Tariff/SmartVA, InterVA, InSilicoVA). Ground truth exists in
principle, and the PHMRC gold-standard study supplies 11,978 deaths (7,580
adults, 1,960 children, 2,438 neonates) across six sites with structured items
*and* free-text narratives. Both individual-level assignment (for civil
registration) and population-level cause fractions (for burden estimation)
matter, so the evaluation metrics (chance-corrected concordance, CSMF accuracy)
are already standardized, which the twin world had to invent for itself.

**Social autopsy** asks about the household's path through illness: recognition,
decision to seek care, reaching care, receiving care (the Three Delays), costs,
and barriers. The integrated VASA tool splices an SA module onto the PHMRC VA
questionnaire. Here the questions are about *behavior under constraint*, which
is exactly where the mega-study's stereotyping and ideological distortions bite.

## Disability weights

### 1. Simulated cognitive testing of lay descriptions

The GBD 2013 hearing-loss and spinal-cord-injury revisions show that a few
dozen words of description can double a weight. Every new or revised state
therefore deserves a comprehension test, and GBD has never had the budget to
cognitively interview 235 descriptions in five languages. Sturgis, Roberts and
Robinson (Survey Futures WP15, 2026) find that a guided
simulated-cognitive-testing prompt detects 75% of deliberately embedded item
flaws with a small false-positive rate, complements expert review, and costs a
few dollars per 20 items. Twin-2K-500 personas add a dimension WP15 lacks:
readers who differ in numeracy, education, need for cognition, and depression
score. Concretely: for each lay description, have each twin paraphrase it,
name the functional domains it evokes (mobility, pain, cognition, affect,
social participation), and rate its severity; flag descriptions where twins
with low crystallized-intelligence scores paraphrase differently from those
with high scores, where the evoked domains miss the intended ones, or where
two descriptions collapse into the same paraphrase. This is cheap enough to
run on every candidate description before any human sees it.

### 2. Twins as a prior for choosing which pairs to ask

With 235 states there are ~27,000 possible pairs; GBD surveys randomized pairs
and relied on volume. Paired-comparison scaling is most informative where the
probability of preferring one state is near 0.5, and a twin panel can produce a
full prior ordering for free before fieldwork. Contaminated or not (the model
has surely seen GBD tables), a prior is only a prior: use it to allocate human
comparisons toward near-ties and toward new or revised states, and let the
humans supply the data. The under-dispersion finding (twin SD ≈ 0.6 of human SD)
matters here: twins will overstate how decisive each pair is, so treat the
twin-implied probabilities as shrunk toward 0.5 before designing.

### 3. Heterogeneity hypotheses the GBD surveys could not test

The DW surveys collected a handful of covariates per respondent. Twin-2K-500
personas carry ~500. Asking the twins whether simulated DWs shift with risk
aversion, time preference, depression, numeracy, or religiosity yields
hypotheses at zero marginal cost, ranked by effect size, for a subsequent human
module (see #5). The mega-study's finding that twins are closer to
demographics-only twins than to humans means any heterogeneity the twins show
is largely *demographic stereotype*, so treat it as a list of things to check,
not findings. Representation bias cuts the same way: the household surveys in
Tanzania and Bangladesh are the populations the twins represent worst.

### 4. The twin-human gap as a measurement of "what lived valuation adds"

Hyper-rationality is a specific, testable hazard for DW: twins should produce
more transitive comparisons than humans, rank states by clinical severity, and
give unrealistically consistent answers to the population health equivalence
questions (which are numeric, and twins ace numeric questions). Rather than
lament this, measure it. Fit the probit to twin responses and to human
responses on the same pairs and compare state by state. States where the twins
and the public disagree most (my guesses: stigmatized mental disorders, chronic
pain, disfigurement, infertility) are the states where a medical-severity prior
is least adequate, i.e. exactly the states that justify running human DW surveys
at all. Human intransitivity and response noise versus twin consistency also
gives a direct estimate of how much of the DW survey signal is "knowledge" and
how much is "valuation." That is a paper, not a pretest.

### 5. A DW module in the next Twin-2K wave

The single most useful concrete step: propose a 10-minute DW module (15 paired
comparisons and 3 population health equivalence items per respondent, GBD
wording) for a future wave of the panel, or as a sub-study in a second
mega-study. It would produce the first DW responses linked to a 500-question
persona, would let #3 and #4 be done with humans instead of twins, and would
give the twin community a value-elicitation benchmark, a genre absent from the
19 current studies. Adding a short self-rated-health and chronic-conditions
block to the persona battery at the same time would let DW researchers finally
ask whether experience of a state changes its valuation, with enough covariates
to say why.

### 6. New-state and translation triage

When a new health state enters GBD, or descriptions are translated for a new
survey country, a twin panel offers a same-day sanity check: does the new
description land where the epidemiologists expect relative to its neighbors,
and does the translated version land where the source did? Disagreements are
cheap to resolve by rewriting before fieldwork. This is the low-stakes version
of #2 and #1 and could run as a routine step in the GBD cycle.

## Verbal autopsy

### 7. PHMRC as the "Twin-2K-500 of deaths"

The mega-study's core trick is: persona = answers to waves 1–3, task = predict
wave 4. PHMRC already has that shape. Persona = a decedent's demographic
block, a subset of the VA items, and the narrative; task = predict the held-out
items. A "decedent-respondent twin" that predicts held-out item responses at
some accuracy, compared against XGBoost on the same inputs (the mega-study's
own baseline design) and against an empty-persona LLM, tells you how much
information each item carries given the others. Two uses follow directly:
instrument shortening (items a twin predicts near-perfectly from the rest are
redundant; SmartVA's short form was built from Tariff item importance, and this
is an independent criterion), and imputation of items missing from legacy VA
datasets so newer algorithms can run on them. The decision-level validation is
the important part: run Tariff or InSilicoVA on twin-imputed versus actual
items and compare cause assignments and CSMF accuracy. If the causes agree, the
imputed items were safe to drop from the interview.

### 8. Synthetic respondents will be too coherent, and that is measurable

LLMs already read VA data well. In Sierra Leone (6,939 deaths), GPT-5 on
narratives reached chance-corrected concordance 0.71 against physician coding
versus 0.44 for InterVA-5 and InSilicoVA on the questionnaire, and CSMF
accuracy 0.90 versus 0.74–0.79. In the CHAMPS network (3,129 under-five deaths
with MITS-confirmed causes), GPT-4o found 46% of malaria deaths versus 30% and
23% for InSilicoVA and InterVA-5. LAVA on PHMRC shows accuracy rising with
narrative length (49% under 250 characters, 63% over 1,000). The twin paradigm
reverses the direction: instead of reading the respondent, *generate* the
respondent. The mega-study predicts how that fails: hyper-rational twins will
produce textbook symptom clusters, know the cause they are describing, and
never contradict themselves. Real relatives forget, conflate, and fill gaps.
So VA data from twins would make any algorithm look better than it is. The
test is simple and worth running once: train Tariff on twin-generated PHMRC
respondents, test on real PHMRC, and report the drop. That number is the
"coherence distortion" of synthetic VA data, and it bounds every downstream
use.

### 9. The CHAMPS study already built a proto-twin and nobody evaluated it

To feed GPT-4o, the CHAMPS analysis reconstructed ("contrived") free-text
narratives from the structured WHO 2016 items. That is a decedent twin
generating text from a checklist. PHMRC has real narratives next to real
items, so the fidelity of such reconstruction is directly measurable with the
mega-study's toolkit: distributional distance between synthetic and real
narratives on length, symptom mentions, temporal detail, hedging; and
downstream cause-assignment agreement. If the synthetic narratives are too
clean, the CHAMPS sensitivity gains are partly an artifact of feeding the LLM a
tidier input than any field interviewer produces.

### 10. Pretesting instrument logic with respondents who never tire

The 2022 WHO VA instrument is an XLSForm with hundreds of items and deep
relevance conditions; the SwissTPH/WHO-VA issue tracker has reports of
questions made unreachable by unsatisfiable skip logic. A panel of simulated
respondents, each assigned a cause and a relationship to the decedent, driven
through the instrument by a simulated interviewer, will traverse branches no
pretest ever reaches and time every path. Local adaptations (Lagos,
Pakistan's VASA, and every new civil-registration rollout) could be regression
tested this way before training a single interviewer. This use is immune to
the individuation problem: it needs coverage, not fidelity.

### 11. Sparing bereaved families the pretest rounds

VA pretesting means interviewing people about a recent death, repeatedly, to
fix wording. Every question that can be caught by a synthetic respondent is one
fewer bereaved family asked to relive an illness for a draft. This is the one
place where the ethical argument for twins is not merely cost. It argues for
routing *all* early-stage VA and SA wording revisions through simulated
testing first, reserving human pretests for the residual.

### 12. HDSS sites already hold the persona battery

Twin-2K-500 spent 145 minutes per person building personas. Health and
demographic surveillance sites hold years of person-level rounds (household
composition, socioeconomic status, pregnancies, illness episodes, care use) on
every resident, and they already run VA on every death. A decedent twin built
from the HDSS record, asked to predict the VA symptom profile, gives a
data-quality check the field has never had: a VA whose reported symptoms are
far from what the person's history predicts is either a surprising death or a
poor interview. Speculative, but the data exist and are linked.

## Social autopsy

### 13. Twins are the wrong instrument for care-seeking content

The mega-study's twins were more trusting of people and institutions, more
accepting of algorithmic decision-making, and moved toward demographic
stereotypes as persona detail increased. Translated to SA: twins of poor rural
households would over-report timely formal care-seeking, under-report
traditional care and cost barriers, and attribute delays by demographic
caricature. Some of that resembles social-desirability bias in real SA
responses, which makes it more dangerous, not less, because the twin would look
plausible. Keep twins out of SA content priors. Use them for what #10 and #11
cover: burden and logic testing of the integrated VASA interview (which runs
1–2 hours), order effects between the VA and SA sections (does asking about
delays first bias symptom reporting, or the reverse), and wording revision.

### 14. Representation is not a prompt-engineering problem

Every finding above is on a US English online panel, and the silicon-sampling
literature reports poor subgroup recovery in non-English settings. If twins are
ever to be more than logic-testers for VA/SA, the panel has to be built where
the deaths are. A Twin-2K-500-style deep panel at one or two HDSS sites,
consented for re-contact, with items written for that context, is the
prerequisite. #12 suggests the surveillance rounds already supply most of it.

## What to copy regardless of domain

### 15. The evaluation discipline

Always report the empty-persona LLM, the demographics-only twin, and a random
baseline alongside the full twin; report SD ratio and distributional distance,
not accuracy alone; report the human-training-set size at which a simple model
matches the twin ("this twin is worth ~100 humans"). For survey development add
one more metric the mega-study lacks: *decision agreement*. Did the twin-based
choice (drop this item, reword that description, ask this pair) match the
choice you would have made from human pretest data? That is the only metric
that matters for design use, and it can be estimated retrospectively wherever a
human pretest already happened.

### 16. Measure test-retest first

The dataset paper could say twins reach 87.7% of the human test-retest ceiling
only because wave 4 re-asked 88 items two weeks later. DW paired comparisons
and VA items rarely have published test-retest figures. Any twin exercise
should start by measuring the human ceiling, because a twin that beats a
respondent's own two-week consistency is over-fitting to noise.

### 17. Open-source work this repo invites (this repo's theme)

- **Port the simulation module forward.** The mega-study repo lacks
  `text_simulation/`; an older version lives in
  `tianyipeng-lab/Digital-Twin-Simulation`. What needs writing is the
  per-question JSON-stub formatter of supplement S2.3 (plus constant-sum, rank
  and side-by-side types), a config-driven prompt assembler, and the
  responses-to-CSV step. The exact prompts and twin outputs for all 13,299
  person-studies are on Hugging Face, so the port can be regression-tested
  against them. File an issue upstream first; the authors may have a copy.
- **An XLSForm/ODK adapter** producing the repo's JSON template format, so WHO
  VA and VASA instruments flow through the same harness as Qualtrics surveys.
  A few hundred lines; unlocks #7–#11.
- **`parse_qsf.py` should `mkdir -p` its output directory.** Trivial.
- **A GBD-style DW module as a `.qsf`** with a probit-fitting evaluation script
  in `mega_study_evaluation/`, contributed as a 20th study spec. It doubles as
  the proposal for #5.

## Hazards specific to these domains

- **Contamination.** GBD DW tables, the WHO instruments, PHMRC, InterVA's
  probability tables, and the Tariff paper are all likely in training data. A
  twin that reproduces GBD weights may be recalling them. Test on perturbed or
  novel descriptions before trusting any agreement.
- **Legitimacy.** DWs are values. A twin-derived weight, however accurate, has no
  standing. Design only.
- **Privacy.** Real VA records name a death, a household, and often a village.
  The Sierra Leone authors flag sending narratives to external APIs; PHMRC is
  public and anonymized, most CRVS data is neither. Local models or on-premise
  inference are a precondition for anything beyond PHMRC.
- **Non-determinism.** The same GPT model assigned different causes to identical
  records on repeat runs in the Sierra Leone study. Any twin-based design
  decision should be replicated across seeds and models before acting on it.
- **The comfort of the plausible.** The mega-study's central lesson is that
  detailed personas made twins *feel* individuated without making them so.
  A twin pretest that finds no problems is weak evidence of no problems.

## Ranking by leverage-per-effort

- **Quick win:** #1 (simulated cognitive testing of lay descriptions) and #10
  (skip-logic traversal of the WHO VA instrument). Days of work, dollars of API
  spend, no human subjects, immediately useful.
- **Scientifically deepest:** #4 (the twin-human gap by health state as a
  measure of what valuation adds beyond knowledge) and #7/#8 (PHMRC decedent
  twins with held-out items and the coherence-distortion number).
- **Compounds:** #5 (a DW module plus health-status items in the next Twin-2K
  wave) and the XLSForm adapter in #17. Each makes every other idea cheaper.
- **Do not do:** twin-derived disability weights, or SA content priors from
  twins (#13).
