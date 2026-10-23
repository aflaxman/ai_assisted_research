# Adding a model structure

The engine is declarative: a `Structure` is a list of `Compartment`s and a list
of `Transition`s, and every part of the pipeline (cohort simulation, the JAX
likelihood, the transmission embedding, the tests) reads those lists
generically. Adding a structure is writing one builder function and registering
it.

## 1. Write the builder

```python
from tbmodels.core import Compartment, Structure, Transition, const, foi

def structure_two_stage() -> Structure:
    comps = [
        Compartment("S", "susceptible"),
        Compartment("Lf", "latent"),           # fast latent
        Compartment("Ls", "latent"),           # slow latent
        Compartment("I", "disease", infectious=True, rel_inf=1.0,
                    symptomatic=True, bact_positive=True),
        Compartment("Rec", "recovered"),
        Compartment("Dead", "dead"),
    ]
    transitions = [
        Transition("S",  "Lf", foi(),             "infection"),
        Transition("Lf", "I",  const("d"),        "fast progression"),
        Transition("Lf", "Ls", const("e"),        "stabilisation"),
        Transition("Ls", "I",  const("c"),        "slow reactivation"),
        Transition("I",  "Rec", const("self_cure"), "self-cure"),
        Transition("I",  "Dead", const("tb_death"), "TB death"),
    ]
    return Structure(
        name="TwoStage", family="classic",
        compartments=comps, transitions=transitions,
        param_names=["d", "e", "c", "self_cure", "tb_death"],
        entry=lambda p: {"Lf": 1.0},      # a newly infected cohort starts in Lf
        disease_states=("I",),
    )
```

### Compartments

`kind` is one of `susceptible`, `latent`, `disease`, `cleared`, `recovered`,
`dead`. Disease compartments carry the flags the rest of the pipeline reads:

| flag | meaning | used by |
|---|---|---|
| `band` | `minimal` / `subclinical` / `clinical` (spectrum) | subclinical fraction, converter targets |
| `infectious` | contributes to the force of infection | transmission |
| `rel_inf` | relative infectiousness (float, or a parameter name) | transmission FOI |
| `symptomatic` | found by passive care-seeking | transmission detection |
| `bact_positive` | detectable by a sputum test (Xpert/smear/culture) | prevalence, screening |

### Transitions

A transition is `(src, dst, rate)`. The rate is a function
`(params, lam, t) -> float`:

- `const("c")` — a fixed parameter, `params["c"]`.
- `foi(coeff=1.0)` — `coeff * lam`, the force of infection (for `S -> ...`).
- any `lambda p, lam, t: ...` — e.g. a split `lambda p, lam, t: lam * p["a"]`,
  or a time-since-infection hazard `lambda p, lam, t: p["x1"] * max(1.0, t) ** p["x2"]`.

### Entry

`entry(params)` returns where a newly infected cohort lands at t=0 in cohort
mode, as `{compartment: probability}` summing to 1. In transmission mode the same
distribution is the target of the force-of-infection flow out of `S`.

## 2. Register it

Add the builder to `CLASSIC_BUILDERS` in `tbmodels/classic.py` (or
`SPECTRUM_BUILDERS` in `tbmodels/spectrum.py`). `registry.all_structures()` then
includes it, and the parameterized tests cover it automatically.

## 3. What works immediately, and what needs a line

- **Cohort simulation and cumulative incidence** work with no further code: the
  scipy engine builds the generator from the transition list. Constant rates use
  `expm`; a rate that depends on `t` is integrated with LSODA.
- **The differentiable likelihood** needs a JAX forward path. Constant-rate
  structures already work through `jaxcore.jax_first_passage` /
  `jax_occupancy_from`. A new *time-varying* hazard needs either a closed-form
  cumulative in `jaxcore` (as for K and L) or a JAX ODE; otherwise the structure
  can still be simulated and compared, just not NUTS-calibrated through that
  rate.
- **Calibration priors**: `calibrate._sample_param` assigns a prior by parameter
  name. A new parameter name falls through to the family default (half-normal for
  classic rates, uniform for spectrum rates). Add a case there for a bespoke
  prior.
- **Transmission** works for any structure whose compartment flags and `entry`
  are set. Structures whose progression hazard depends on time since infection
  (like K and L) have no autonomous-ODE transmission form and are listed in
  `transmission.TRANSMISSION_EXCLUDED`.

## 4. Check it

```bash
uv run python -m pytest tests/test_structures.py -k TwoStage   # or just run all
uv run python -c "from tbmodels.registry import get; s=get('TwoStage'); print(s.names())"
```

The parameterized tests verify mass conservation, non-negativity, monotone
cumulative incidence, and (for constant-rate structures) that the JAX forward
matches the scipy engine.
