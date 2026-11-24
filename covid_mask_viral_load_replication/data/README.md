# Data sources

| File | Source |
|---|---|
| `SimulatedParameters.txt` | 10,000 simulated within-host parameter sets from the authors' repository for the companion paper, <https://github.com/ashish2goyal/SARS_CoV_2_Super_Spreader_Event> (Goyal, Reeves, Cardozo-Ojeda, Schiffer & Mayer, eLife 2021;10:e63537). Unmodified. Columns: `beta`, `delta`, `k`, `m`, `w` (omega), `de` (delta_E), `E50` (phi), `r`, `q`, `p` (pi), `c`; the `tzero` column is not used by their code. |
| `google_mobility_king_county_2020.tsv` | Google LLC, "Google COVID-19 Community Mobility Reports", <https://www.google.com/covid19/mobility/>, downloaded 24 November 2026 from `Region_Mobility_Report_CSVs.zip` (last modified 18 October 2022). King County, Washington, 15 February to 30 September 2020. Values are percent change from Google's baseline. |
| `paper_fig2_digitized.tsv` to `paper_fig6_digitized.tsv`, `paper_suppfig5_digitized.tsv` | Read off the figures of Goyal et al. (2021), Scientific Reports 11:11838, <https://doi.org/10.1038/s41598-021-91338-5> (CC BY 4.0), by the scripts in `../digitize/`. Each script's docstring says how well its values can be trusted. |
| `paper_fig7_read_by_eye.tsv` | Read by eye off Figure 7 of the same paper; the file's header gives the precision. |
