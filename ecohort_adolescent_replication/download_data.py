"""Download the open (CC0) MNH eCohort analytic datasets from Harvard Dataverse.

These are sibling deposits of the restricted dataset used by Amboko et al. (2026),
PLOS Global Public Health, doi:10.1371/journal.pgph.0004729.
"""
import urllib.request
from pathlib import Path

FILES = {
    # doi:10.7910/DVN/SSVXKY  (Arsenault et al., eCohort paper 1: ANC standards)
    "eCOANC1.dta": 10271859,
    "Codebook_eCOANC1.xlsx": 10391402,
    # doi:10.7910/DVN/Q0YKOT  (Yang, ANC quality and perinatal outcome)
    "ecohort_perinatal_outcome.dta": 13595671,
    "ecohort_perinatal_outcome_long.dta": 13595670,
    # doi:10.7910/DVN/JGAXNG  (Clarke-Deelder, Ethiopia malnutrition/anemia)
    "econut_analyticdata.dta": 12061984,
    "econut_codebook.xlsx": 12061985,
    # doi:10.7910/DVN/AUQ8C8  (Amboko et al. — code and README are open, data is restricted)
    "Amboko_analysis_2025.do": 13417120,
    "Amboko_inferential_analysis_2025.do": 13417118,
    "Amboko_Data_Readme.txt": 13417119,
}

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
for name, fid in FILES.items():
    dest = DATA / name
    if dest.exists():
        print("exists", dest)
        continue
    url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}?format=original"
    print("downloading", name)
    urllib.request.urlretrieve(url, dest)
print("done")
