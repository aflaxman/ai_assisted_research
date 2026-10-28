# Running notes: sodium reduction replication

Working log for the replication of Dhungel et al. (2025). Newest entries at
the bottom. The README holds the write-up; this file holds what I tried, in
order, including the dead ends.

## 2026-10-27 (cloud session, summarized)

- Built the model on the Claude cloud sandbox from GBD 2023 pulled through
  the IHME GBD MCP server. Results: HALY gains 65% to 88% of the paper's
  medians for the seven main scenarios, about half for the two small UK
  programs; mortality reductions about half to two-thirds of the paper's.
- Suspects for the shortfall, from the decomposition: the mortality pathway
  (deaths averted), not the disability pathway.
- Supplementary material was not reachable from the sandbox (ScienceDirect 403).

## 2026-10-27 (cluster session)

Session moved to the IHME cluster. New inputs available: the paper's
supplementary appendix (uploaded by Abie), `vivarium_inputs` and `gbd_mapping`
against the GBD databases, the old `ihmeuw/vivarium_sodium_reduction` repo,
and the recent `vivarium_csu_mace_rct` project for data access patterns.

Question to answer: why are deaths averted smaller in my replication?

### Appendix, first read (mmc1.pdf, 28 pages)

- **Table S3 gives the SBP equation.** Per gram of sodium removed:
  `2.026 + 0.502 * (age - 50)/10 + 0.659 * (SBP - 140)/10 + 2.353 * p_nonwhite`,
  with p_nonwhite = 0 for Australia. My reconstruction from Huang et al. had
  0.371 per decade of age and 0.248 per 10 mmHg, so the paper's age slope is
  1.35x mine and its SBP slope 2.7x mine. At age 70 and SBP 150 the paper
  gets 3.69 mmHg per gram against my 3.0; at 80 and 160, 4.85 against 3.6.
  Deaths are concentrated exactly there, so this is a prime suspect.
- Table S5 confirms the equation is floored at zero (young, low-SBP cells
  show 0.00) and shows about 3.6 mmHg per gram at 70+ and SBP 140 to 149.
- **Sodium TMREL of 2 g/day for the SBP link**: only intake above 2 g/day
  counts, following Trieu et al. Needs checking against the sodium means
  (Table S1): if the paper's baseline intake is higher than GBD 2023's
  2.1 to 4.1 g/day, the absolute reduction (proportion x baseline) is larger
  in the paper for the same proportional cut.
- **Table S6, the 18 diseases**: IHD, ischemic stroke, ICH, SAH, hypertensive
  heart disease, atrial fibrillation, stomach cancer, non-rheumatic calcific
  aortic valve disease, other cardiomyopathy, CKD (T1 diabetes, T2 diabetes,
  hypertension, glomerulonephritis, other), rheumatic heart disease, aortic
  aneurysm, endocarditis, peripheral artery disease. Mine has 13 of these
  (no T1-diabetes CKD, calcific aortic valve, other cardiomyopathy, rheumatic
  heart disease, endocarditis, since GBD 2023 no longer attributes them to SBP).

### Baseline exposures: the paper's Tables S1 and S2 versus GBD 2023

Table S1 (Trieu et al.) gives sodium means by sex, age and SEIFA quintile;
averaging the five quintiles, men run from 4.4 g/day at 25 to 29 down to
3.1 at 80+, women from 3.2 to 2.6. GBD 2023's exposure for Australia, which
my first artifact used, is 3.6 to 2.4 for men and 2.8 to 2.2 for women. The
proportional reductions in Table 1 therefore remove 20 to 45% more sodium
in the paper than in my model, most of the gap at the ages where the deaths
are. Table S2's SBP means are close to GBD's for men but 5 to 7 mmHg lower
for women at older ages, with SDs of 12 (age 25) to 21 (age 85); my assumed
SDs were 11 to 20, so that part was fine.

Decision: switch the sodium and SBP exposure inputs to the paper's tables
(quintiles averaged with equal weight, since SEIFA quintiles are equal-size
by construction) and keep GBD for the disease rates and relative risks.
Transcribed tables are in `data/paper/`.

### Cluster resources

- `vivarium_inputs` 7.2.4 + `gbd_mapping` 5.0.4 in the
  `vivarium_csu_alzheimers_artifact` conda env, pointed at GBD 2023
  (release 16). Pulls for Australia are running in the background into
  the scratchpad; each measure takes 10 to 60 s against the database.
- GBD 2023 attributes high SBP to 12 causes (same set as GBD 2021); the
  paper's extra five are GBD 2019 (release 6) outcomes only.
- Slurm: partitions `all.q` and `long.q`, account `proj_simscience`
  (from the MACE RCT sbatch scripts). This node has 4 CPUs.
