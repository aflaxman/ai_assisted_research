## Model vs GBD 2023 (United States), comparison year 2023

| Quantity | MYRIAD replication (PWID cohort) | GBD 2023, United States (95% UI) | Note |
|---|---|---|---|
| People who inject drugs | 3,694,500 (2018 input, Bradley et al.); 3,371,107 alive in model, Jan 2023 | Injected drug use exposure, 2022: 1.06% of males, 0.48% of females (all ages) ≈ 2,570,317 people | GBD's exposure series is the input to its drug-use risk model; the count is my arithmetic (proportion x GBD population). |
| Opioid use disorder, people | 2,770,730 in the high-risk (OUD) state, Jan 2023 (input: 45.1% of PWID in 2018 = 1,666,220) | Prevalence 2023: 6,990,364 (6,058,906–7,956,232); 2018: 5,893,492 (5,205,231–6,611,306) | The model covers only the injecting subset; GBD counts all OUD. The 2023 model paper's own calibration denominator was 22.8M people misusing opioids (2015), 35% with OUD. |
| Drug use disorders (all drugs), people | 2,848,880 in high- plus medium-risk states (medium = illicit non-OUD use) | Prevalence 2023: 12,796,839 (11,594,263–14,140,490) | Medium-risk also includes non-opioid injection (e.g. methamphetamine), which GBD counts under amphetamine/cocaine use disorders. |
| Overdose deaths per year | 48,813 among PWID in 2023 (paper S-B3 estimate: 46,902 = 45.1% of all overdose deaths) | Drug use disorders deaths 2023: 95,333 (76,054–115,274); opioid use disorders: 72,697 (57,750–89,669) | GBD codes US overdose deaths to drug use disorders. Model PWID deaths / GBD all-drug deaths = 0.51, close to the paper's 45.1% injection multiplier. |
| Overdose deaths per year, 2018 | 30,243 (paper target 30,226) | Drug use disorders deaths 2018: 62,894 (51,426–76,441); opioid use disorders: 47,530 (38,875–57,962) |  |
| Annual overdose death probability, people with OUD | 1.17% in 2025 (input, high-risk state); 1.66% in 2023 (S-B3) | OUD deaths / OUD prevalence 2023 = 72,697 / 6,990,364 = 1.04% (my ratio of two GBD estimates) | The model's rate is for injecting people with OUD, a higher-risk subset of GBD's OUD population. |
| People with HIV | 182,630 among PWID, Jan 2023 (input: 5.12% of PWID in 2018) | HIV prevalence 2023: 1,799,956 (1,079,296–2,552,080); share of HIV YLDs attributable to drug use: 6.9% (5.7–8.5%) | The paper's introduction cites 'nearly 100,000 PWID live with HIV', half its own input (5.12% x 3.69M = 189,000). GBD's drug-use PAF applied to its prevalence would give about 124,197 (my arithmetic). |
| New HIV infections per year | 3,423 among PWID in 2023 (input: 0.112% annual incidence, CDC IDU + MSM/IDU categories) | HIV incidence 2023, all transmission: 58,512 (29,026–90,645) | GBD publishes no drug-use-attributable incidence. CDC's surveillance estimate for all new infections (~32,000) sits at the low end of GBD's interval. |
| HIV deaths per year | 1,090 among PWID in 2023 | HIV deaths 2023: 5,676 (5,351–5,993); attributable to drug use: 376 (310–474) (PAF 6.6%) | The model's HIV deaths among current PWID are about 3x GBD's drug-use-attributable HIV deaths and a fifth of all US HIV deaths. |
| People with chronic HCV | 1,549,087 among PWID, Jan 2023 (input: 43.5% of PWID in 2018) | Chronic hepatitis C including cirrhosis, prevalence 2023: 4,615,743 (3,766,638–5,624,341) | CDC's viremic estimate (about 2.4M, 2017-2020) is below GBD's interval; the model's PWID carry a third of GBD's count. |
| New HCV infections per year | 23,090 among PWID in 2023 (input: 1.28% annual incidence among susceptible) | Acute hepatitis C incidence 2023: 210,180 (179,849–244,237) | GBD's acute HCV incidence is far above CDC's estimated acute infections (about 67,000 in 2022); the model's 23,000 is current PWID only. |
| HCV deaths per year | 5,803 among PWID in 2023, rising to 11,510 by 2034 | Total burden related to hepatitis C, deaths 2023: 37,964 (33,634–42,777) (chronic HCV incl. cirrhosis 27,512, liver cancer due to HCV 10,446); attributable to drug use: 33,769 (28,456–38,580) (PAF 88.9%) | GBD attributes nearly all US HCV deaths to drug use (ever injected), but most occur in older former PWID outside the model's active-IDU cohort: the model's current-PWID HCV deaths are 15% of GBD's total. |
| Endocarditis per year | 7,310 hospitalisations among PWID in 2023 (4.8% of severe infections); about 424 deaths at the 5.8% in-hospital fatality input | Endocarditis incidence 2023: 67,728 (59,676–77,844); deaths: 9,450 (8,160–10,666) | PWID account for about a tenth of GBD's US endocarditis incidence in this comparison, in the range of published injection-associated shares. |
| Skin and soft-tissue infections per year | 131,287 hospitalisations among PWID in 2023 | Bacterial skin diseases (cellulitis, pyoderma) incidence 2023: 11,418,188 (11,125,111–11,729,716) episodes, all severities | Not comparable: GBD counts all episodes, the model only hospitalisations. Osteomyelitis and CNS abscess have no GBD cause. |

### Background mortality: model input vs GBD life table

Five-year probabilities of death per 1,000 (model: cumulated from the supplement's monthly, cause-deleted table; GBD: 2023 abridged life table, all causes).

| Age | Model background q(x), male | GBD all-cause q(x), male | Model, female | GBD, female |
|---|---:|---:|---:|---:|
| 20-24 | 6.5 | 6.9 | 2.3 | 2.6 |
| 25-29 | 7.5 | 9.0 | 3.0 | 3.5 |
| 30-34 | 8.4 | 11.5 | 4.1 | 5.1 |
| 35-39 | 9.8 | 13.8 | 5.5 | 6.8 |
| 40-44 | 12.5 | 17.1 | 7.8 | 9.3 |
| 45-49 | 18.4 | 21.4 | 12.1 | 12.3 |
| 50-54 | 29.7 | 29.5 | 18.8 | 17.8 |
| 55-59 | 44.8 | 43.8 | 27.5 | 26.7 |
| 60-64 | 64.3 | 64.3 | 38.6 | 40.5 |
| 65-69 | 87.6 | 89.3 | 56.9 | 57.6 |

### All-cause mortality in the simulated cohort

| Year | Model cohort all-cause deaths per 1,000 person-years | of which other causes | Approximate mean age (40 at entry) |
|---|---:|---:|---:|
| 2018 | 14.2 | 2.8 | 40 |
| 2023 | 21.5 | 4.0 | 45 |
| 2025 | 18.1 | 4.8 | 47 |
| 2034 | 28.8 | 9.1 | 56 |
