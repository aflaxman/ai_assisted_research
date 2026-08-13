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
