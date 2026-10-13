# TB natural-history model-structure comparison

A reproducible comparison of tuberculosis natural-history **model structures**:
the 12 classic latent-progression structures catalogued by Menzies et al. (2018)
versus the spectrum models of Richards et al. (2023) and Horton et al. (2023),
which resolve disease into minimal, subclinical and clinical states.

The question: do the two families differ in (a) fit to natural-history cohort
data, (b) the subclinical fraction they imply, and (c) the impact they predict
for community-wide screening?

See `PLAN.md` for the full design, data inventory and the resolved design
decisions, and `docs/likelihood_blocks.html` for why the two families are fit to
overlapping (not identical) blocks of data.

## Layout

```
tbmodels/            the package
  core.py            declarative Structure/Transition engine (scipy; cohort + transmission)
  classic.py         the 12 Menzies structures A-L + shared exit block
  spectrum.py        the 4 spectrum variants (Horton/Richards and ablations)
  registry.py        all structures in one place
  jaxcore.py         differentiable (JAX) forward model for calibration
  likelihood.py      the three-block cohort likelihood
  calibrate.py       numpyro NUTS calibration + pointwise log-likelihoods
  transmission.py    embed a structure in a population transmission model
  gbd_targets.py     GBD / UNAIDS fitting targets
  data.py            loaders with provenance
scripts/             calibrate_cohort, fit_transmission, simulate_act3, build_db, analyze
data/vendored/       redistributable inputs (epidemery MIT, Menzies Dataverse CC0)
data/raw/            git-ignored; populate with `uv run python fetch_data.py`
data/gbd/            GBD 2023 and UNAIDS target series
data/derived/        posteriors, fits, simulations (generated)
docs/                likelihood-block explainer; adding-a-structure guide
results/             the comparison tables, figures and the DuckDB file
report.qmd           the Quarto write-up
```

## Reproduce

```bash
uv sync --all-extras
uv run python fetch_data.py                 # third-party inputs at pinned versions
uv run python -m pytest                      # engine tests
uv run python scripts/calibrate_cohort.py --all      # cohort posteriors
uv run python scripts/fit_transmission.py --all      # GBD fits (5 countries)
uv run python scripts/simulate_act3.py --all         # ACT3 screening
uv run python scripts/analyze.py                     # comparison tables + figures
uv run python scripts/build_db.py                    # assemble the DuckDB file
uv run quarto render report.qmd                      # the write-up
```

## Adding a model structure

A structure is a list of compartments and a list of transitions; nothing in the
engine hard-codes any particular model. To add one, write a builder that returns
a `Structure` and register it.

```python
from tbmodels.core import Compartment, Structure, Transition, const, foi

def structure_my_variant() -> Structure:
    comps = [
        Compartment("S", "susceptible"),
        Compartment("L", "latent"),
        Compartment("I", "disease", infectious=True, rel_inf=1.0,
                    symptomatic=True, bact_positive=True),
        Compartment("Rec", "recovered"),
        Compartment("Dead", "dead"),
    ]
    transitions = [
        Transition("S", "L", foi(),            "infection"),   # foi() = force of infection
        Transition("L", "I", const("c"),       "progression"), # const("c") reads params["c"]
        Transition("I", "Rec", const("self_cure"), "self-cure"),
        Transition("I", "Dead", const("tb_death"), "TB death"),
    ]
    return Structure(
        name="MyVariant", family="classic",
        compartments=comps, transitions=transitions,
        param_names=["c", "self_cure", "tb_death"],
        entry=lambda p: {"L": 1.0},     # where a newly infected cohort lands (cohort mode)
        disease_states=("I",),
    )
```

Then add it to the registry (`tbmodels/classic.py` `CLASSIC_BUILDERS` or
`tbmodels/spectrum.py` `SPECTRUM_BUILDERS`, or call it directly). That is the
only change needed:

- **Cohort fit** works immediately: `simulate_cohort`, `cumulative_incidence`,
  and the JAX forward model read the transition list generically. Constant-rate
  structures use `expm`; a time-varying rate (a `rate` function that depends on
  its third argument `t`) is integrated numerically, and if it has a closed-form
  cohort first-passage, add it to `jaxcore` for a differentiable likelihood.
- **Transmission** works immediately for any structure with an `entry`
  distribution and the compartment flags set (`infectious`, `rel_inf`,
  `symptomatic`, `bact_positive`); `TransmissionModel` adds the susceptible
  pool, force of infection, reinfection, detection and demography.
- **Tests**: `tests/test_structures.py` is parameterized over the registry, so a
  new structure is automatically checked for mass conservation, non-negativity
  and monotone cumulative incidence.

Compartment `kind` is one of `susceptible`, `latent`, `disease`, `cleared`,
`recovered`, `dead`. A rate is `const("name")`, `foi(coeff)`, or any
`lambda p, lam, t: ...`. Full detail in `docs/adding_a_structure.md`.

## Sources

- Menzies NA, et al. *Lancet Infect Dis* 2018;18:e228 (PMC6070419) and the
  replication data, Harvard Dataverse doi:10.7910/DVN/STRA2P (CC0).
- Richards AS, et al. *Lancet Glob Health* 2023 — `natural_history_tb_disease`.
- Horton KC, et al. *PNAS* 2023;120:e2221186120 — `Inf-Dis` (CC BY 4.0).
- Emery JC, Dodd PJ, et al. — `subclinical_transmission` (MIT).
- Tiemersma EW, et al. *PLoS ONE* 2011;6:e17601.
- Marks GB, et al. *N Engl J Med* 2019;381:1347 (ACT3 trial).
- GBD 2023 via the IHME MCP; ART coverage from the World Bank (UNAIDS estimates).
```
