# SAMHSA OUD Data Exploration

Notes on SAMHSA datasets relevant to Opioid Use Disorder (OUD) research, compiled March 2026.

## Key Datasets

### 1. NSDUH (National Survey on Drug Use and Health)

**Best for:** OUD prevalence estimates in the general population, MOUD receipt rates.

Annual household survey of civilian, noninstitutionalized U.S. population aged 12+.

**OUD-relevant variables:**
- Heroin use (past year, lifetime)
- Prescription opioid misuse (hydrocodone, oxycodone, tramadol, codeine, morphine, prescription fentanyl, buprenorphine, oxymorphone, hydromorphone, methadone, meperidine)
- Illegally made fentanyl (IMF) use (added 2022)
- OUD diagnosis (DSM-5 criteria)
- Receipt of Medications for OUD (MOUD)
- Age at first use, frequency, route of administration
- Co-occurring mental health conditions
- Demographics (age, sex, race/ethnicity, income, education, employment)

**Key 2024 finding:** Among 4.8M people aged 12+ with past-year OUD, 17.0% (818,000) received MOUD.

**Coverage:**
- National estimates released annually (most recent: 2024 data released 2025)
- State-level estimates from pooled 2-year data (2023-2024 expected early 2026)
- Substate estimates from pooled 3-year data (2023-2025 expected ~summer 2027)

**Access:** Public-use files (PUFs) at https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/datafiles. Interactive analysis via https://www.datafiles.samhsa.gov/analyze-data. No public API — download only (SAS, SPSS, CSV).

**Caveats:**
- **2020 methodology break:** Shift from in-person-only to multi-mode (in-person + web). Estimates from 2020+ not comparable to 2019 and earlier. Cannot pool across the break.
- **IMF measurement gap:** IMF questions come after the SUD assessment section, so SUDs arising solely from IMF use cannot be identified.
- **Evolving opioid definitions:** Pre-2024 "opioids" = heroin + prescription pain relievers. Starting 2024, "opioids" = heroin + prescription opioids (narrower subset).
- **Population exclusions:** Excludes incarcerated, unsheltered homeless, active-duty military — groups with potentially high OUD rates.
- **Self-report bias.**

---

### 2. TEDS (Treatment Episode Data Set)

**Best for:** Treatment admissions/discharges, demographic profiles of people entering treatment, long time series (since 1992).

Administrative dataset tracking admissions (TEDS-A, since 1992) and discharges (TEDS-D, since 2000) from substance use treatment facilities that receive public funding.

**OUD-relevant variables:**
- Primary, secondary, tertiary substance at admission (heroin; other opiates/synthetics)
- Route of administration (injection, inhalation, oral, smoking)
- Planned medication-assisted opioid therapy (MAOT)
- Frequency of use, age at first use
- Service type (outpatient, residential, detox, MAOT-specific)
- Demographics, criminal justice referral, prior treatment episodes
- Discharge data: length of stay, reason for discharge

**Key 2022 finding:** 18.3% of admissions were for heroin; 11.3% for other opiates/synthetics. 9.7% received MAOT outpatient services.

**Coverage:** National and state-level. CBSA variable in PUFs. TEDS-A from 1992, TEDS-D from 2000.

**Access:** https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set/datafiles

**Caveats:**
- **Admission-based, not person-based.** One person admitted twice = two records. Cannot track individuals in PUFs.
- **State variation in reporting.** States define "admission" differently; cross-state absolute count comparisons are invalid.
- **Incomplete coverage.** Only publicly funded/state-licensed facilities. Misses private treatment, office-based buprenorphine prescribing outside these systems.
- **State exclusions.** States with counts below 50% of prior 3-year average are dropped. In 2022: Delaware, Oregon, Washington, West Virginia excluded.

---

### 3. N-SUMHSS (formerly N-SSATS + N-MHSS)

**Best for:** Treatment facility availability, MOUD capacity, state-level treatment infrastructure.

Annual point-in-time survey of all known substance use and mental health treatment facilities in the U.S. Merged from N-SSATS and N-MHSS in 2021.

**OUD-relevant variables:**
- Whether facility is a certified Opioid Treatment Program (OTP)
- Medications offered: methadone, buprenorphine, injectable naltrexone
- Client counts receiving each medication
- Facility characteristics (ownership, setting, payment types)
- Special population programs (pregnant women, criminal justice, etc.)

**Coverage:** National and state-level profiles (all 50 states, DC, territories). N-SSATS: 2000-2020; N-SUMHSS: 2021+. Point-in-time snapshot (typically March).

**Access:** https://www.samhsa.gov/data/data-we-collect/n-sumhss-national-substance-use-and-mental-health-services-survey/datafiles

**Caveats:**
- Facility-level, not client-level (counts clients but no individual records)
- 2021 survey redesign may affect comparability with pre-2021 data
- Voluntary participation
- Point-in-time snapshot misses seasonal variation

---

### 4. MH-CLD (Mental Health Client-Level Data)

**Best for:** Co-occurring OUD + mental health conditions.

Client-level records from State Mental Health Agencies on individuals in publicly funded mental health treatment.

**OUD relevance:** Limited but complementary. Includes up to three mental health diagnoses per client plus substance use characteristics.

**Access:** https://www.samhsa.gov/data/data-we-collect/mh-cld-mental-health-client-level-data/datafiles/2023

**Caveats:** Only covers SMHA-funded facilities. Data swapping for confidentiality. Primarily a mental health dataset with limited OUD detail.

---

## Cross-Cutting Issues

| Issue | Details |
|-------|---------|
| No public API | All datasets: bulk download only (SAS, SPSS, CSV) or interactive DAS tool |
| Data lag | 1-2 years between collection and public release |
| Treatment gap | Most datasets miss private/commercial insurance treatment and office-based prescribing |
| No linkage in public data | PUFs strip identifiers; cannot link individuals across datasets |
| Fentanyl undercounting | IMF new to NSDUH (2022); TEDS lumps fentanyl under "other opiates/synthetics" |
| Population exclusions | NSDUH excludes incarcerated/homeless/military; TEDS only covers public treatment |
| State reporting variability | TEDS and MH-CLD depend on state systems that vary in completeness and definitions |

## Quick Reference: Dataset by Research Question

| Research Question | Best Dataset(s) |
|-------------------|-----------------|
| OUD prevalence in general population | NSDUH |
| Treatment admissions/discharges | TEDS |
| Treatment facility availability & MOUD capacity | N-SUMHSS |
| Demographic profiles of people entering treatment | TEDS |
| MOUD receipt rates at population level | NSDUH |
| State-level treatment infrastructure | N-SUMHSS + TEDS |
| Co-occurring mental health + OUD | MH-CLD + NSDUH |
| Long time series trends | TEDS (since 1992), NSDUH (with 2020 break caveat) |

## Sources

- [SAMHSA Data Portal](https://www.samhsa.gov/data/)
- [NSDUH National Releases](https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/national-releases)
- [NSDUH Data Files](https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/datafiles)
- [Supplemental NSDUH Opioid Tables](https://www.samhsa.gov/data/report/supplemental-nsduh-opioid-tables)
- [TEDS Data Files](https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set/datafiles)
- [N-SUMHSS Data Files](https://www.samhsa.gov/data/data-we-collect/n-sumhss-national-substance-use-and-mental-health-services-survey/datafiles)
- [MH-CLD Data Files](https://www.samhsa.gov/data/data-we-collect/mh-cld-mental-health-client-level-data/datafiles/2023)
- [SAMHSA Interactive DAS](https://www.datafiles.samhsa.gov/analyze-data)
- [Strengths and Weaknesses of Existing Data Sources for Opioid Research (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6971390/)
- [Data Needs in Opioid Systems Modeling (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8061725/)
- [CDC Overdose Prevention Data](https://www.cdc.gov/overdose-prevention/data-research/facts-stats/index.html)
