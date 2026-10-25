# AI-Assisted Research Project

This repository contains notes and experiments for AI-assisted open source development work.

## Structure

Use subdirectories for each research project, to keep things separated, e.g.

- `mpld3_notes/` - Development environment and PR testing for mpld3/mplexporter

## Environment

I would like to keep the environments easy to create and isolated, e.g. for mpld3_notes I will be

- Running on WSL (Windows Subsystem for Linux)
- Use `python -P` to avoid path shadowing issues when working inside cloned repos
- Browser files open via `explorer.exe` on WSL


## Conventions

- Use `uv` for Python package management
- Commit messages include the Claude Code attribution footer
- **Quickstart sections**: Should be simple and direct - choose ONE recommended path, not multiple options. Don't force users to make decisions in quickstart; save alternatives for detailed instructions.

- for mpld3_notes: Test scripts follow the pattern `test_pr<NUMBER>_<description>.py`

## camdl (compartmental model DSL)

[camdl](https://github.com/vsbuffalo/camdl) is a DSL + compiler + inference stack
for stochastic compartmental models (OCaml frontend `camdlc` → IR JSON → Rust
backend `camdl`). Used in `camdl_replication/` and
`yf_brazil_climate_replication/`. Notes below are from camdl 0.1.0+e8219d1
(Oct 2026) and cover things that cost real time to rediscover.

### Getting the binaries

Both toolchains are needed: `camdlc` compiles `.camdl` → IR, `camdl` simulates and
fits. `./install.sh` in the camdl repo does everything, but in a sandbox it is
usually faster to build the two halves separately:

```bash
# Rust half (~6 min, 4 cores) — gives `camdl`
cd rust && cargo build --release --workspace --bins

# OCaml half — gives `camdlc`. Needs opam + OCaml >= 5.2.
opam init --bare --no-setup --yes --disable-sandboxing   # no bubblewrap in containers
opam switch create camdl 5.2.0 --yes
opam install dune menhir yojson fmt --yes
printf 'let value = "%s"\n' "$(tr -d '[:space:]' < ir/VERSION)" \
  > ocaml/lib/ir/ir_version_generated.ml     # Makefile normally generates this
cd ocaml && dune build                       # test targets need alcotest; ignore those failures
cp ocaml/_build/default/bin/camdlc.exe ~/.local/bin/camdlc
cp rust/target/release/camdl              ~/.local/bin/camdl
```

`camdl docs <topic>` is version-matched and offline — prefer it to memory.
`docs/dsl-cheatsheet.md` and `docs/fit-toml.md` in the repo are the fastest read.

### Gotchas that cost time

- **No inverse-count dimension.** The dimension lattice has only `[1]`, `[P]`,
  `[T]`, `[1/T]`, `[T^-1]`, `[P/T]`, `[P*T^-1]`. A mass-action term written
  `beta * S * I` (no `/N`) is E300 and *cannot* be expressed by annotating `beta`.
  Write it per-capita against a fixed reference count:
  `beta * S * (I / n_ref)` with `beta = beta_raw * n_ref`. Exactly equivalent.
- **`let` bound to a numeric literal is dimensionless**, even with a unit literal
  (`let m = 4e8 'count` does not carry `[P]`). For a constant that must carry a
  dimension, declare it as a `count` parameter and fix it in `[fixed]`.
- **`count`-kind initial values are rounded to whole numbers.** `y0 = 0.6` and
  `y0 = 1.0` give identical runs. This silently breaks any scaling argument that
  pushes an initial count below 1.
- **Incidence observation streams need `covers`** (E350). On a weekly model,
  `closing_at(time, 1 'weeks)` puts week *k*'s window at `(k-1, k]`;
  `ending_on(time, 1 'weeks)` anchors to the last included *day* and lands the
  boundary at `k + 1/7`, which is then "not a recorded output time".
- **`emit_schedule` is separate from `covers`.** Without it a stream is fit-only
  and `simulate --obs` refuses it.
- **`flow_<name>` columns are per-interval, not cumulative.** With
  `--output-every 1`, row *t = k* already holds week *k*'s incidence; do not diff
  them (that silently yields negative "incidence"). Row *t = 0* is a zero-width
  interval — drop it.
- **`ivp = true` is gone**; the `[estimate]` field is now `perturb_only_at_t0`
  (an IF2 concept, irrelevant on the `ode` backend). `camdl_replication/`'s
  configs predate this.
- **Pick `dt` exact in binary** (`0.0625`, `0.125`) so observation-period
  boundaries land on recorded output times. `1/7` does not.
- **Use `log_uniform` priors for scale parameters.** `uniform` over `[1e-12, 100]`
  puts essentially every Latin-hypercube start in the top decade, and the fit
  never finds a small optimum.
- **A chain-agreement FAIL is not automatically a bad fit.** With broad priors most
  starts land in hopeless regions. Check `chain_results.tsv` and see whether the
  *best* chains agree before believing the verdict either way.
- **`mle_params.toml` has a nested `[provenance]` table.** Read only top-level
  scalars as parameters, or `chain`/`log_likelihood` leak in and `simulate` errors
  with "unknown parameter".

### Backends and algorithms

| algorithm | backend | gives |
|---|---|---|
| `if2` / `pgas` / `pmmh` / `pfilter` | `chain_binomial` | stochastic-process likelihood |
| `nl-sbplx` / `nl-bobyqa` | `ode` | deterministic MLE (`chains` = LHS starts) |
| `mh` / `nuts` | `ode` | posterior on the deterministic ODE marginal |

For large populations where demographic noise is negligible, the `ode` backend is
much faster and is the right analogue of a published least-squares fit. `nuts`
additionally needs a differentiable model and fixed `rk4` (not adaptive `rk45`).

### Workflow

Runs are content-addressed under `results/`, so an identical re-run is a cache hit
and nothing is overwritten. `camdl check` type-checks (including every rate's
dimension) before anything runs — always run it first. Browse with
`camdl list --kind fit`, `camdl fit summary <dir>`, `camdl cat <id>`.

Drive scenario sweeps from Python by shelling out to `camdl simulate` with a
generated params TOML (see `yf_brazil_climate_replication/yf_model.py`); pass
`--output-dir` to a temp dir for throwaway runs so the project store stays clean.

## NHANES Data Analysis

**Always apply NHANES survey weights.** NHANES uses a complex, multistage probability
sample with oversampling of subgroups, so unweighted statistics are *not*
nationally representative and can be biased. Weighting is rarely cosmetic — it
routinely shifts prevalences and distribution shapes (e.g., in
`nhanes_cap_lsm/`, weighting changed below-threshold F4 prevalence from 0.9% to
0.5%).

- **Pick the weight that matches the rarest component used.** For exam (MEC)
  variables such as elastography or labs, use the MEC weight, not the interview
  weight. For the 2017–2020 pre-pandemic file, that is `WTMECPRP` (interview:
  `WTINTPRP`); for two-year cycles it is `WTMEC2YR` / `WTINT2YR`. Subsample
  files (e.g., fasting labs) carry their own special weights — use those.
- **2017–2020 is a special combined cycle.** Use the pre-pandemic `WTMECPRP` /
  `WTINTPRP` weights built for the ~3.5-year period; do not pool two-year
  weights yourself. When combining multiple two-year cycles, divide each cycle's
  weight per the NHANES analytic guidelines instead.
- **Weight every estimate**, including histograms (`weights=`), KDEs
  (`gaussian_kde(..., weights=)`), and prevalences (ratio of weighted sums).
- **Design-based variance** (standard errors, CIs) additionally needs the design
  variables `SDMVPSU` and `SDMVSTRA` via Taylor linearization (or a survey
  package). Suppress/flag small cells by the *unweighted* count.

## Technical Blog Post Guidelines

When writing technical blog posts for healthyalgorithms.com:

### Project Structure

1. **Create a new subdirectory** for each blog post (e.g., `simple_fuzzy_checker_application/`)
2. **Use `uv` to set up a Python environment** in each subdirectory for isolated dependencies
3. **Put the blog draft in `README.md`** in the subdirectory

### Content Structure

1. **Keep it simple** - Focus on clarity over complexity
2. **Start with a hook** - Begin with a minimal, concrete code example that demonstrates the core topic
3. **Include a TL;DR section** - Provide quick takeaways at the beginning
4. **Include a graphic** - Add a visualization (animation, diagram, or plot) to illustrate the concept
5. **Provide runnable code** - Make the code accessible via:
   - Colab notebook (add badge: `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](link)`)
   - Binder notebook
   - Self-contained scripts
6. **Separate concerns** - Keep simulation/implementation code separate from test code
7. **Link to specific code** - Reference code with pattern `file_path:line_number` or GitHub permalink

### File Organization

Each blog post directory should include:
- `README.md` - The blog post content
- `requirements.txt` - Python dependencies for `uv`
- Implementation files (e.g., `simulation.py`)
- Test files (e.g., `test_simulation.py`)
- Jupyter notebook (e.g., `tutorial.ipynb`)

### Writing Style & Tone

Based on healthyalgorithms.com established voice:

1. **Accessible yet technical** - Balance rigor with conversational explanations
2. **Use first person** - Write with "I" to create personal connection
3. **Pragmatic positioning** - Frame tools/methods as useful rather than revolutionary
4. **Metaphorical headers** - Use engaging section titles (e.g., "The Problem:", "An Answer:")
5. **Question-driven narrative** - Open with a fundamental challenge readers face
6. **Progressive complexity** - Start simple, build to advanced applications

### Post Structure Pattern

Follow this proven narrative arc:

1. **Visual hook** - Start with an animation, graphic, or minimal code example
2. **TL;DR section** - What readers will learn, get, and the core approach
3. **Problem statement** - Articulate the fundamental challenge
4. **Solution introduction** - Present the method/tool/approach
5. **Concrete example** - Walk through a specific, complete implementation
6. **Pedagogical elements**:
   - Code snippets with explanations
   - File-by-file breakdowns
   - Links to GitHub permalinks for code context
7. **Adaptation guidance** - "How to use this in your work"
8. **Challenges/Exercises** - Engage readers beyond passive consumption
9. **Further reading** - External resources and references

### Interactive Elements

- **Colab/Binder badges** - Make code immediately runnable
- **Animated visualizations** - GIFs showing the concept in action
- **External links** - Papers, GitHub repos, related tools
- **Challenge sections** - Prompt readers to extend or experiment

### Prose Style: Strunk and White Principles

Apply these timeless writing principles for clarity and impact:

1. **Omit needless words** - Cut ruthlessly. "The question as to whether" → "Whether"
2. **Use the active voice** - "The bug was caught by the test" → "The test caught the bug"
3. **Put statements in positive form** - "Not honest" → "Dishonest"; avoid "not un-" constructions
4. **Use definite, specific, concrete language** - Replace vague terms with precise examples
5. **Place emphatic words at the end** - Save the punch for the sentence's conclusion
6. **Express coordinate ideas in similar form** - Parallel structure aids comprehension
7. **Keep related words together** - Minimize distance between subject and verb, modifier and modified
8. **Use orthodox spelling** - Maintain credibility through correct conventions
9. **Make the paragraph the unit of composition** - One paragraph = one topic
10. **Begin each paragraph with a topic sentence** - Let readers know where they're going
11. **Use figures of speech sparingly** - Technical writing needs clarity over ornamentation
12. **Avoid a succession of loose sentences** - Vary structure to maintain engagement
13. **Do not break sentences in two** - Two weak sentences are worse than one strong sentence
14. **Avoid fancy words** - "Use" beats "utilize"; "help" beats "facilitate"
15. **Be clear** - When choosing between clarity and style, always choose clarity

**Application to technical blog posts:**
- Replace passive constructions: "is calculated by" → "calculates"
- Front-load value: Put the insight before the explanation
- Cut filler: "In order to" → "To"; "It should be noted that" → delete
- Strengthen verbs: "makes use of" → "uses"; "is in violation of" → "violates"
- Concrete examples: Not "some bugs," but "three directional bias bugs"

## Replicating epidemic-model papers with camdl

[camdl](https://github.com/vsbuffalo/camdl) ([intro](https://vincebuffalo.com/blog/introducing-camdl/))
is a DSL + compiler + inference stack for stochastic compartmental models
(OCaml frontend, Rust engine), developed at the Institute for Disease Modeling.
It is a strong default for replicating compartmental-model papers: the compiler
dimension-checks every rate, runs are content-addressed and cached, and failure
modes (particle degeneracy, non-convergence, discretization dependence) are
diagnosed rather than silent. Two worked replications live in
`policy_behavioral_camdl/` (behavioral-feedback SIR) and the
`claude/replicate-paper-camdl-*` branch (Cui 2026 SEIRD + He et al. 2010 measles).

### Install (no sudo, from source)

Not on PATH by default. Rust ships in the base image but OCaml/opam do not, so
build from source — takes ~15–25 min (OCaml 5.2.0 compiler + vendored nlopt +
Rust engine):

```bash
git clone --depth 1 https://github.com/vsbuffalo/camdl   # public; reachable via proxy
cd camdl && NO_SANDBOX=1 ./install.sh                     # → ~/.local/bin/camdl
export PATH="$HOME/.local/bin:$PATH"
```

Run it in the background and keep working; the binary appears before `make test`
finishes. Prereqs (make, git, curl, tar, cmake ≥ 3.13) are already present.

### DSL structure

A `.camdl` model file has these blocks (see `docs/dsl-cheatsheet.md` and
`docs/camdl-language-spec.md` in a clone — read the cheatsheet first):

```camdl
time_unit = 'days                      # 'weeks/'months/'years; add origin=date(...) to anchor
compartments { S, I, R }
let N = S + I + R                       # let CAN reference state, params, forcings, covariates
parameters { beta : rate in [0.1,1.5]  # kinds: rate probability count positive real duration instant
             rho  : probability in [0,1] }
forcing { policy : interpolated 'ratio { data="p.tsv" time_col="time" value_col="policy" method=linear } }
transitions { infection : S --> I @ beta * S * I / N        # rate must be dimension P·T⁻¹ (E300)
              recovery  : I --> R @ gamma * I }
init { S = N0 - i0 ; I = i0 ; R = 0 }
observations { cases { columns { time:time, cases:count }
                       projected = incidence(infection)      # or a compartment
                       emit_schedule = every 1 'days
                       cases ~ poisson(rate = projected + 0.001) } }   # also normal/neg_binomial/beta/beta_binomial
quantities { R0 = final(beta/gamma) ; peak_day = time_of_max(I) }      # derived, non-scored
simulate { from = 0 'days ; to = 120 'days }
```

Units are first-class: literals carry them (`5 'days`, `0.1 'per_day`,
`100 'count`, `1.0 'ratio`), and the checker rejects dimensionally wrong rates.
Synthetic/textbook models stay **unanchored** (no `origin`, bare numeric times);
real-calendar data uses `origin = date(...)` with `time_unit = 'days`/`'weeks`.

### Expressing behavioral / time-varying / noisy dynamics

- **Time-varying covariate** (policy, seasonality, births): a `forcing` block —
  `interpolated` from a tsv, `periodic`, or `sinusoidal`; reference as `name(t)`.
  A forcing can take a `lag` (duration). Bake smooth ramps (e.g. delayed
  behavioral adaptation) into the covariate itself — rate expressions cannot
  introduce auxiliary ODE state.
- **State-dependent feedback** (alarm/behavior responding to prevalence): inline
  a function of the compartment in the rate, or via a `let` — e.g. a Hill term
  `endog_delta / (1 + (x0/(I + 1 'count))^nu)`. Prevalence `I` is a natural
  ~`1/gamma`-day smoother of recent incidence, so it stands in for
  moving-average-incidence inputs while staying fittable.
- **Process noise / heterogeneity**: wrap a rate in `overdispersed(rate, sigma2)`
  for Gamma noise on the force of infection (σ² dimensionless, E308). This is the
  idiomatic way to model heterogeneous compliance / extra-demographic noise.
- **Reactive policies** (`reactive_interventions {}` with `when sum_observed(...)`)
  read observed incidence over a window — but they run in **forward simulation
  only** and error under inference. For a *fittable* behavior-responds-to-cases
  model, use the prevalence-driven `let` form instead.
- **Non-exponential dwell times**: `E --> I via erlang(stages=3, rate=sigma)` or
  `via hyper_erlang(branch(...), ...)`. `stages` is a structural literal (not
  fittable); `mean`/`rate` are.

### Workflow (commands)

```bash
camdl check model.camdl                                    # compile: units + dimensions
camdl simulate model.camdl --params p.toml --backend chain_binomial --dt 1.0 \
      --seed 42 --obs-only cases.tsv                        # synthetic observed series
      # --param name=val overrides; --replicates N ensemble; -o traj.tsv = state path
camdl fit run fit.toml --label run --seed 3                 # staged: if2 scout → pgas posterior
camdl fit summary results/fits/<dir>/                       # R-hat, gate verdict, MLE table
camdl fit predict --fit <dir> --stream cases                # predicted-vs-observed ribbon
camdl simulate model.camdl --draws posterior --fit <dir> -n 300 --obs-only pp.tsv
camdl survey model.camdl --fit fit.toml                     # LHS identifiability landscape
```

A `fit.toml` declares `[model]`, `[data.observations]`, `[estimate]` (per-param
`bounds`/`start`/`prior`; `ivp=true` for initial-value params), `[fixed]`, and
`[stages.*]` (`algorithm`, `backend`, `chains`, `particles`, `iterations`/`sweeps`).
Posterior draws land at `results/fits/<dir>/**/draws.tsv`.

### Practical gotchas

- The complete-data likelihood is very informative for large populations, so
  posteriors are sharp — a hand-rolled Metropolis needs proposals scaled to the
  MLE SEs or it sits at 0% acceptance (camdl's PGAS avoids this).
- Under-resourced particle filters get killed by an ESS-collapse watchdog with an
  actionable message (more particles / tighter bounds); the convergence gate
  fails small scout fits (R̂ threshold + decibans) rather than passing them.
- When the full paper is paywalled, replicate the framework in camdl on
  simulated data (parameter recovery + the paper's qualitative claims), and say
  so plainly — same honesty rule as the Python replications above.
