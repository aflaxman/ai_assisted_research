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

# Open (CC-BY 4.0) Zenodo deposits by Gebeyaw Molla Wondim: the Ethiopia eCohort with PHQ-9 at
# every timepoint. The ABO file is a near-complete Ethiopia M1-M5 wide extract (816 women, 4326 vars).
ZENODO = {
    "wondim_antenatal_depression_long.dta": "https://zenodo.org/records/19205479/files/Dataset_v2.dta?download=1",
    "wondim_antenatal_depression_code.do": "https://zenodo.org/records/19205479/files/Statistical%20analysis%20code.do?download=1",
    "wondim_ethiopia_m1m5_wide_abo.dta": "https://zenodo.org/records/20730905/files/Dataset_antenatal%20depressive%20symptoms%20and%20ABOs.dta?download=1",
    "wondim_perinatal_depression_long.dta": "https://zenodo.org/records/19119541/files/Dataset%20for%20perinatal%20depressive%20symptoms%20and%20IHRQoL_long%20format.dta?download=1",
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
for name, url in ZENODO.items():
    dest = DATA / name
    if dest.exists():
        print("exists", dest)
        continue
    print("downloading", name)
    urllib.request.urlretrieve(url, dest)
print("done")
