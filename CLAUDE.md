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

## Writing in my voice

Any prose written for me to sign or send (blog posts, READMEs, notebook
markdown, data notes, memos, email drafts, referee responses) should be in my
voice: a researcher with a mathematics and computer science background who
writes about statistical methods for global health, for smart readers who may
be outside the field. Do not fall back on a generic "clear academic" register;
the two are not the same. Before drafting, name the type of piece and the
audience or venue, and set the register dial (below) accordingly. The targets
here were measured on the Introduction and the eight solo-authored chapters of
*An Integrative Metaregression Framework for Descriptive Epidemiology* (2015),
about 25,000 words, so hit the numbers rather than interpreting adjectives.

### Stance

- **Own the choices in the first person singular.** "I have used," "I prefer,"
  "I recommend," "which I call." First-person singular runs about 7 per 1,000
  words, five times the rate of "we." Use "we" only for something writer and
  reader are doing together on the page ("we turn to an example now") or for
  actual collaborators. Never hide a judgment behind "it was decided" or "the
  literature suggests."
- **Pragmatist, not purist.** Theory is judged by whether it helps in practice.
  State the principled objection fairly and in full, then say what you actually
  did and why the compromise is acceptable. The characteristic attitudes: a
  method is "simply a method of convenience"; "something is better than
  nothing"; a tool is "too convenient not to be used"; models are built up until
  they are "just complicated enough."
- **Put candor about weaknesses next to the recommendation**, not in a
  limitations section. Every method gets its benefit and its drawback, usually
  in one sentence: "X has the best balance of A and B, but cannot cope with C."
  Say plainly when a result is "not very good," an approach is "crude," a
  feature is "undesirable," or a term you coined "promises to be extremely
  confusing!" Never sell.
- **Respect the reader's time and intelligence.** Say what you are skipping and
  why ("this would take us off our course"), point to where the full treatment
  lives, and warn when something is hard ("first-time readers should proceed
  knowing it will be easier the second time around"). Reassure without
  condescending.
- **Dry humor, about one wry line per section.** Deadpan statement of the
  obvious ("The unfortunate truth is that no one lives forever") or
  self-deprecation ("Unfortunately, I've seen the data"). Never at the reader's
  expense, never whimsy for its own sake.
- **An occasional personal anecdote makes an abstraction concrete.** A prior
  belief explained by what smoking looked like in Pittsburgh bars versus
  Seattle; a student who could not see why Bayes' rule deserved anyone's name.
  At most one per piece, and it must do explanatory work.

### Architecture

1. **Open in plain language.** Frame the problem in a paragraph or two before
   any notation. A short etymological or historical aside is a favored hook
   (where "spline" comes from; Mahalanobis and the harvest weight of plants;
   al-Khwarizmi; Ulam's solitaire), but only if it lands on the point within
   three sentences.
2. **Announce the plan, then follow it.** "I will consider five: …" followed by
   a short list.
3. **Motivating example before general theory, and make it real.** A named
   disease, a named study, actual numbers with units (782 data points; an
   850-fold range; 40 times higher; 1,000 times longer to compute).
4. **Simplest model first; add complexity one step at a time.** Name the
   specific failure that forces each step, and show the failure (a figure, a
   number) before proposing the fix.
5. **Formalism, then intuition.** Give the equation, define every symbol in
   words immediately, then "the intuition behind it is simple:" and tell the
   story in plain terms (n people were tested, k tested positive). Reverse the
   order when the equation looks worse than it is: "Although this equation may
   look imposing, it has a simple representation as…"
6. **Signpost.** Forward and backward references to specific sections are
   frequent and exact.
7. **Close with "Summary and future work."** Two or three sentences on what was
   developed and which option you prefer, with its tradeoff. Then concrete open
   directions, each carrying an honest weight: "promising," "worthy of
   additional attention," "sure to be a fruitful area," "will certainly have its
   own computational challenges."

### Sentences

- **Long but unpadded.** Median about 23 words, mean about 27; one sentence in
  five runs past 35 words, and short punchy sentences are rare (under 8% are 10
  words or fewer). Length comes from subordination (commas, "which," "where," a
  mid-sentence "however" or "but"), never from stacked adjectives. Every clause
  adds a qualification, a tradeoff, or a pointer.
- **Punctuation.** Commas and parentheses (word-bearing asides about 2.5 per
  1,000 words); semicolons rare (under 1 per 1,000); dashes almost absent (0.2
  per 1,000). Colons introduce equations and lists. One exclamation mark per
  chapter, for real surprise or a joke. Rhetorical questions are rare and
  arrive in a run of three ("Why minimize the squared residuals? Why not…? Why
  not…?").
- **Cohesion by demonstrative.** Roughly one sentence in ten opens with "This"
  pointing at the previous idea ("This provides a way to critique the model").
  "However" opens sentences freely and also sits mid-sentence between commas.
- **Hedge with frequency, not confidence.** "Often," "sometimes," "usually,"
  "in practice": words about how often something happens in the data. Almost
  never "perhaps," "arguably," "it could be argued." Almost no intensifiers or
  sincerity markers ("truly," "genuinely," "importantly"; "clearly" appears
  about once per 4,000 words). Uncertainty is a fact about the data, stated
  plainly, not a wobble in the prose.
- **Terminology.** Coin names for methods and say so ("which I call rate
  models," "hard-soft constraints"). Put jargon in quotation marks on first use,
  then drop the quotes. Flag misleading names ("its strange name comes from…,"
  "the somewhat opaque term…").
- **Impersonal constructions for the machinery, first person for the choices.**
  "Can be written as," "can be interpreted as," "is deferred to Section 8.7"
  describe what the mathematics does; "I have used," "I prefer" describe what
  the author did.

### Do not

- Sell. No "novel," "powerful," "robust," "state-of-the-art," or
  "comprehensive" as puffery; no superlative you cannot measure.
- Perform confidence or sincerity ("it is crucial to note," "genuinely,"
  "importantly").
- Fragment argument into bullets. Bullets enumerate models, steps, or
  properties; the argument happens in paragraphs.
- Bury limitations at the end or soften them into "may have some limitations."
  Name the drawback next to the claim.
- Use dashes as the default rhythm, or chain semicolons.
- Write short, choppy "impact" sentences. This voice trusts a long sentence to
  carry a qualified thought.
- Invent examples or numbers when real ones exist. If a value is not known, say
  so.

### Calibration lines (my own)

- "For the purposes of this book, however, it is simply a method of convenience."
- "…make them more complicated until they are just complicated enough."
- "Although this equation may appear opaque, the intuition behind it is simple:"
- "On the other hand, something is better than nothing."
- "This approach is crude, however, and leaves much room for further work."
- "When the model is more flexible than the data, it is hard to fit…"
- "…and if they show the opposite, that is interesting."
- "Proceeding down this path with the proposed terminology promises to be
  extremely confusing!"

### One rewrite, to fix the register

Generic: "Our novel Bayesian framework robustly handles sparse and noisy data.
Importantly, extensive simulations demonstrate superior performance across all
metrics, highlighting its potential to significantly advance the field."

In this voice: "The model I have used handles sparse, noisy data better than
the alternatives I compared it to, in the sense that its uncertainty intervals
cover the held-out observations about as often as they should (Table 2). This
is a lower bar than it sounds. The simulation that generated those observations
was designed by me, and a model tends to look good on data that share its
assumptions. A comparison on data withheld from systematic review would be more
convincing, and it is a direction for future work."

### Register dial

The targets above come from a monograph. For shorter pieces (op-ed, blog post,
memo, email, README, notebook markdown), keep the stance intact: first person,
verdict with tradeoff in the same sentence, one wry line, an honest note on what
remains open. Bring median sentence length down toward 18 words, drop the
section scaffolding and the "Summary and future work" header, and use at most
one equation, or none.

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

Write in the voice described in "Writing in my voice" above, at the blog-post
setting of the register dial. In addition, based on healthyalgorithms.com's
established voice:

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

Apply these timeless writing principles for clarity and impact. They operate
inside the voice described in "Writing in my voice"; where they pull apart
(Strunk and White's brevity against this voice's long, subordinated sentences),
keep the sentence long and cut only the padding.

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
