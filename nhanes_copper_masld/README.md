# NHANES copper and MASLD: recovered write-ups and a pointer to the missing code

> **TODO (Abie): search the home laptop for the August 2026 copper/MASLD work
> and add it to this directory if you find it.** The code was never pushed to
> GitHub. Look for a local git repo containing `analysis/` notebooks `01`–`07`
> (executed, at commit `1ef987c`), the modules `measurement.py`, `causal.py`,
> and `survey.py`, and a file named `ukb_replication_plan.md`. See
> [Finding the prior work](#finding-the-prior-work) below for where to look.

## What this is

In August 2026 I asked Claude to review a set of Stata analyses of **dietary
copper intake and hepatic steatosis in NHANES**. The working hypothesis: copper
homeostasis is disturbed in steatotic liver disease, and lower dietary copper
may contribute to steatosis (MASLD, metabolic dysfunction-associated steatotic
liver disease). The review grew into a Python re-analysis (notebooks 01–07)
and a plan to replicate the findings in UK Biobank.

The work was done in a Claude Code CLI session run through Remote Control on my
laptop, titled "Review Stata scripts and documentation" (2026-08-14 to
2026-08-17). That session had no GitHub repository attached. A full search on
2026-09-17 of this repo (every branch and PR head),
`ai_assisted_us_health_data_analysis`, `towards_us_crl_estimates`, and
`vivarium_research` found no trace of the code. The only surviving outputs are
two private Claude Artifacts, transcribed to Markdown here:

| File in this directory | Source artifact | Date |
|---|---|---|
| [`nhanes_review_and_plan.md`](nhanes_review_and_plan.md) | [Copper & MASLD](https://claude.ai/artifact/9SL27yht5LmW1kbRoo1Epv) | 2026-08-14 |
| [`ukb_replication_plan.md`](ukb_replication_plan.md) | [UKB Copper Replication](https://claude.ai/artifact/54aBXLtw8WFFgMhn3V9sf8) | 2026-08-17 |

## Key findings from the review (short version)

- Energy-adjusted dietary copper is inversely associated with steatosis in two
  independent modalities: elastography CAP (β ≈ −16 dB/m per mg/1000 kcal,
  P ≈ 0.01, n = 4,137) and DXA visceral fat (after age/sex adjustment).
- The apparent contradictions dissolve on inspection: age is a suppressor for
  the visceral-fat association, FIB-4 carries age in its formula (adjusted
  β = 0), BMI adjustment is over-adjustment, and serum copper behaves as an
  acute-phase marker rather than an intake proxy.
- **None of the Stata estimates are usable as-is.** Both `svyset` calls omit
  `pweight`, so every number is unweighted (see the NHANES weighting rules in
  the top-level `CLAUDE.md`). Several models also condition on the outcome
  (CAP ≥ 248 / ≥ 268 restrictions), and the exposure has skewness ≈ 10.
- The UK Biobank plan pre-specifies seven hypotheses (H1–H7). The decisive one
  is H3, whether a copper-specific effect survives diet-quality adjustment,
  which NHANES was underpowered to settle (≈ −4 to −5 dB/m-equivalent, n.s.).

## Finding the prior work

On the laptop (WSL), try in this order:

```bash
# 1. Transcripts of local Claude Code sessions. The folder name encodes the
#    working directory the session ran in, which is where the repo should be.
grep -rl -i copper ~/.claude/projects/ | head

# 2. The replication plan's repo copy has a distinctive name.
find ~ -name ukb_replication_plan.md -not -path '*/.git/*' 2>/dev/null

# 3. Any repo that contains the commit the plan cites.
find ~ -maxdepth 5 -name .git -type d 2>/dev/null | while read g; do
  git -C "$g/.." cat-file -e 1ef987c 2>/dev/null && echo "$g"
done

# 4. Or resume the session by title from the directory it ran in.
claude --resume
```

If the repo was on the Windows side, repeat steps 2 and 3 under
`/mnt/c/Users/<you>/`.

The Stata inputs the review covered live on the IHME shared drive (treat as
read-only): `Cu_CAP_Aug2026.log` (10 Aug 2026),
`CuNHANES_DexaFat_Aug2026.log` (11 Aug 2026), and `CuNHANES_CAP_Notes.docx`.

## What to add once found

- `analysis/` notebooks 01–07 (executed) and the supporting modules
  (`measurement.py`, `causal.py`, `survey.py`), plus their `pyproject.toml` and
  `uv.lock`, following the layout of `nhanes_cap_lsm/`.
- The cohort flow table and the corrected (weighted) estimates behind the
  "NHANES anchor" column of the UK Biobank plan.
- The repo copy of `ukb_replication_plan.md`, replacing the transcription here
  if they differ.
- Do **not** commit raw NHANES files. Keep to code, derived tables, and figures.
