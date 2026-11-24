# Replicating "Using Probabilistic Cellular Automata to Model the Transmission of an Emerging Infectious Disease of Amphibians" (Duffus et al., Pathogens 2026)

An attempt to replicate, in Python, every quantitative result in:

> Duffus ALJ, Standridge JP, Bartlett PL, George JC (2026). Using Probabilistic
> Cellular Automata to Model the Transmission of an Emerging Infectious Disease
> of Amphibians—A Preliminary Model. *Pathogens* 15(8):827.
> https://doi.org/10.3390/pathogens15080827 (CC BY 4.0)

The paper simulates a ranavirus outbreak in a pond of 100 common frogs on a
10×10 grid, with two viral strains (ulcerative U and hemorrhagic H, plus a
combined state C), local Moore-neighborhood transmission, and per-iteration
disease mortality ω.

## TL;DR

1. **The tables replicate.** An independent NumPy reimplementation reproduces
   every entry of Table 1 and Table 2 within Monte Carlo error. Running the
   authors' own supplementary code (with its hard-coded seed) reproduces the
   ω = 0.15 and ω = 0.25 rows of Table 1 *bit-for-bit* (e.g., 10.0222 / 37.6406 /
   9626 exactly).
2. **One table entry is a single-digit typo.** Table 1 reports mean time to
   pond extinction of **37.79** iterations at ω = 0.175, breaking the otherwise
   monotone trend. The authors' own code with its shipped seed gives
   32.7866… → **32.79** for that cell — while the rest of the row (9.97,
   9368 ponds) matches the paper digit-for-digit. A "2" became a "7".
3. **Figure 4 was not made with the parameters in its caption.** The caption
   says ω = 0.15, but under ω = 0.15 the mean Combined curve peaks at ~37
   frogs and ponds are typically fully dead by iteration ~38. The published
   figure shows Combined peaking at ~70 and only ~89 dead at iteration 40.
   Digitizing the published curves and scanning ω shows the figure matches
   **ω = 0.06** (see `results/figure4_forensics.png`): the authors' own code
   at ω = 0.06 gives a Combined peak of 69.3 at iteration 10 (published:
   ~69.4), an Ulcerative peak of 11.1, a Hemorrhagic peak of 4.2, and a
   max-across-ponds Combined band reaching 88 — each matching Figures 4 and 7.
   The paper's in-text claim that combined infections "peak at about 70
   individuals" describes that lower-mortality run, not the headline scenario.
4. **A quirk in the authors' transmission code shapes the state mix.** In
   `test_infect`, a susceptible frog that catches *both* strains from a single
   combined-state neighbor in one pass ends up in state U, not C (the
   ulcerative branch overwrites the hemorrhagic one). Fixing that quirk
   changes the state composition substantially (U peak drops from ~13 to ~2,
   C peak rises from ~37 to ~43 at ω = 0.15) but barely moves the headline
   extinction statistics.

## Quickstart

```bash
cd ranavirus_pca_replication
uv venv .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python replicate_results.py          # tables + figures, ~2 min
.venv/bin/python figure4_forensics.py          # the omega=0.06 comparison
```

## What the model is

Each cell of a 10×10 grid holds one frog in state S, U, H, C, or D. One frog
(position (8,8)) starts in state C; the rest are susceptible. Each iteration:

1. Every living, not-fully-infected frog checks its ≤8 Moore neighbors in a
   fixed order. Each neighbor carrying a strain the focal frog lacks triggers
   an independent Bernoulli draw (σ_U = 0.65, σ_H = 0.75). Newly infected
   frogs cannot transmit until the next iteration (synchronous update).
2. Every infected frog — including those infected this very iteration — dies
   with probability ω (0.15 in the headline scenario).

Dead frogs never transmit and are never replaced. The paper's contact rate ψ
is folded into the transmission probabilities (μ = ψσ), which is how Table 2
varies it.

## Files

| File | Purpose |
|---|---|
| `ranavirus_ca.py` | Vectorized NumPy reimplementation (all ponds simulated at once; ~100× faster than the original) |
| `replicate_results.py` | Runs the baseline scenario plus the Table 1 and Table 2 sweeps; writes CSVs and figures to `results/` |
| `figure4_forensics.py` | Overlays curves digitized from the published Figure 4 on replication runs at ω = 0.15 and ω = 0.06 |
| `original/Ranavirus(Jul-13-26).py` | The authors' code, verbatim from Supplementary File S1 |
| `results/table1_replication.csv` | Paper's Table 1 next to replicated values |
| `results/table2_replication.csv` | Paper's Table 2 next to replicated values |
| `results/figure4_replication.png` | Replication of Figure 4 at the caption's ω = 0.15 |
| `results/figure4_forensics.png` | Evidence that the published Figure 4 used ω ≈ 0.06 |
| `results/fig4_digitized.json` | Curve values extracted from the published Figure 4 PNG by pixel color-matching |

## Detailed findings

### Table 1 (mortality sweep) replicates, with one bad cell

Mean iteration until all frogs infected ("All Infected"), mean iteration until
pond extinction among extinct ponds ("All Dead"), and number of extinct ponds
out of 10,000 ("Ponds"), paper vs. this replication (independent NumPy
implementation, different RNG stream):

| ω | All Infected (paper / repl.) | All Dead (paper / repl.) | Ponds (paper / repl.) |
|---|---|---|---|
| 0.075 | 9.84 / 9.84 | 71.12 / 70.83 | 9832 / 9849 |
| 0.100 | 9.95 / 9.96 | 54.41 / 54.47 | 9904 / 9904 |
| 0.125 | 10.01 / 10.02 | 44.39 / 44.45 | 9799 / 9802 |
| 0.150 | 10.02 / 10.02 | 37.64 / 37.67 | 9626 / 9613 |
| 0.175 | 9.97 / 9.96 | **37.79** / **32.89** | 9368 / 9361 |
| 0.200 | 9.74 / 9.75 | 29.39 / 29.33 | 8950 / 8943 |
| 0.225 | 9.41 / 9.44 | 26.85 / 26.63 | 8392 / 8458 |
| 0.250 | 8.84 / 8.87 | 24.76 / 24.71 | 7670 / 7700 |

The published 37.79 at ω = 0.175 sits *above* the ω = 0.150 value in a
strictly decreasing column. Running the authors' own code (fixed seed
28473892) at ω = 0.175 reproduces the row's other entries exactly (9.9696 →
9.97; 9368 ponds) and gives 32.7866… → 32.79 for this cell: the published
value has a "7" where a "2" belongs.

Two other quirks of these statistics, visible in the code:

- "All Infected" is averaged over **all** 10,000 ponds, but ponds where the
  infection dies out before reaching everyone contribute 0 rather than being
  excluded. That, not faster spread, is why the column *falls* as mortality
  rises above 0.15.
- "All Dead" is correctly averaged only over ponds that reached extinction.

### Table 2 (contact-rate sweep) replicates

| ψ | All Infected (paper / repl.) | All Dead (paper / repl.) | Ponds (paper / repl.) |
|---|---|---|---|
| 0.75 | 11.01 / 11.16 | 38.53 / 38.26 | 9234 / 9252 |
| 0.80 | 10.91 / 10.88 | 38.29 / 38.12 | 9395 / 9364 |
| 0.85 | 10.63 / 10.56 | 37.99 / 38.02 | 9453 / 9411 |
| 0.90 | 10.35 / 10.38 | 37.85 / 37.76 | 9472 / 9486 |
| 0.95 | 10.23 / 10.22 | 37.79 / 37.91 | 9587 / 9573 |

### Figure 4 forensics

I digitized the five curves in the published Figure 4 by color-matching pixels
in the article PNG, then scanned ω. Checkpoints (published figure vs.
replication at ω = 0.06 vs. at the caption's ω = 0.15):

| Quantity | Published Fig. 4 | ω = 0.06 | ω = 0.15 |
|---|---|---|---|
| Combined peak | 69.4 at iter ~10 | 69.0 at 10 | 37.2 at 10 |
| Dead at iter 20 | 62.5 | 61.4 | ~97 |
| Dead at iter 40 | 89.3 | 88.9 | ~100 |
| Ulcerative peak | ~11 | 11.5 | ~13 |

As a final check I ran the authors' own code at ω = 0.06 (10,000 ponds, their
seed): mean Combined peak 69.27 at iteration 10, Ulcerative peak 11.14,
Hemorrhagic peak 4.15, and a max-across-ponds Combined count of 88 — matching
the published Figure 4 and the band ceiling visible in Figure 7 (~88).
Figure 7 (Combined with min/max band) shows the same ω = 0.06 curves, so
Figures 4–7 all appear to come from one lower-mortality run, while the tables
and the in-text summary statistics come from the stated parameters. The
figure caption and the "no living frogs left after about 38 iterations" text
are internally inconsistent with the plotted Dead curve, which doesn't reach
~100 until iteration ~80.

### The `test_infect` overwrite quirk

In the authors' `test_infect` (lines 152–191 of the supplementary code), both
strain checks compare against the frog's state *captured at the start of the
call*. When a susceptible frog meets a C neighbor and both Bernoulli draws
succeed, the hemorrhagic branch sets its state to H, then the ulcerative
branch — still seeing "S" — overwrites it to U. Catching both strains in one
encounter therefore yields U, not C. Because σ_H > σ_U, this asymmetry makes
U the dominant single-strain state in all published runs (U peaks ~13, H ~5).
With the natural fix (both draws succeed → C), the composition flips: U peaks
~2, H ~9, C rises ~6 frogs, while time-to-extinction statistics change by
less than one iteration. The published results are thus a faithful record of
the code as written, but the U-vs-H asymmetry the paper displays is partly an
artifact of statement ordering rather than biology.

## Reproducibility notes

- The supplementary code sets `rnd.seed(28473892)`, which is why its output is
  bit-reproducible: running it unmodified except for `pd` gives *exactly* the
  published Table 1 values at ω = 0.15 (10.0222 / 37.6406 / 9626) and ω = 0.25
  (8.8368 / 24.7551 / 7670).
- The shipped code has `pd = 0.250` (the last Table 1 column), not the
  headline 0.15.
- My reimplementation reproduces the process distributionally (different RNG
  stream), simulating all 10,000 ponds as one NumPy array; the full paper
  replication runs in about 2 minutes on one core.
- Article and figures are CC BY 4.0; the supplementary code ZIP is available
  at https://www.mdpi.com/article/10.3390/pathogens15080827/s1.
