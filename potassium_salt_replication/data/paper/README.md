# Published results used as comparison targets

Everything here is transcribed from Huang et al., "Global health outcomes from
replacing dietary sodium chloride with potassium-enriched salt", Nature Health
(2026), https://doi.org/10.1038/s44360-026-00216-9, and its Supplementary
Information.  Counts are per year; the global tables are in millions.

| File | Source | How it got here |
|---|---|---|
| `table1_global_results.csv`, `table1_people_covered.csv` | Table 1 | transcribed from the article's web page |
| `table2_ckd_results.csv` | Table 2 | transcribed from the article's web page |
| `extended_data_table1_sensitivity.csv` | Extended Data Table 1 | transcribed from the article's web page |
| `supp_table3_country_results.csv` | Supplementary Tables 3a to 3f | `paper_targets/parse_supplement.py` |
| `supp_table4_ckd_results.csv` | Supplementary Tables 4a to 4c | `paper_targets/parse_supplement.py` |

The three global tables were transcribed by an automated reader of the article
page and have not been checked by eye against the PDF.  There is one mechanical
check: all thirty cardiovascular and kidney disease cells of Table 1 agree, to
rounding, with the sums of the global rows of Supplementary Table 3, which were
parsed from the supplement's PDF by a script.  Table 2 and Extended Data Table 1
have no such check.  The article is open access under a Creative Commons
Attribution 4.0 licence.

Table 1 carries a footnote saying that its deaths are net of the deaths caused
by hyperkalaemia, and Extended Data Table 1 is titled "net effect".  Scenario
names are this repository's (`constants/metadata.py`), in the paper's order.
