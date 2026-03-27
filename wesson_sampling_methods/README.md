# Combining Venue-Based and Respondent-Driven Sampling with Random Forests

A simulation of the method from Wesson et al. (2025) for merging two
probability-based sampling approaches to survey hidden populations.

## The paper

Wesson P, Graham-Squire D, Perry E, Assaf RD, Kushel M. "Novel methods to
construct a representative sample for surveying California's unhoused
population: the California Statewide Study of People Experiencing
Homelessness (CASPEH)." *American Journal of Epidemiology*. 2025;194(5):1238-1248.
[PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12055459/)

## The problem

Surveying people experiencing homelessness (PEH) is hard. Two standard
approaches each miss part of the population:

- **Venue-based sampling (VBS)** goes to shelters, encampments, and service
  sites. It reaches people who use services but misses farmworkers, DV
  survivors, and others who avoid these venues.
- **Respondent-driven sampling (RDS)** uses peer referral chains. It reaches
  socially connected subgroups but undersamples isolated individuals and
  shelter users.

Neither method alone produces a representative sample.

## The method

Wesson et al. combine VBS and RDS into a single representative sample in
three steps:

1. **Draw both samples independently.** VBS produces the main sample; RDS
   supplements it with hard-to-reach subgroups.

2. **Predict cross-method selection probabilities with random forests.**
   Each participant was sampled by one method, so their selection probability
   under that method is known by design. A random forest trained on
   participant characteristics predicts what their selection probability
   *would have been* under the other method.

3. **Compute overall inclusion probability and weight.** For each person:

       P(included) = P(VBS) + P(RDS) - P(VBS) * P(RDS)

   This is the probability of being selected by at least one method
   (union of two independent events). Each participant is then weighted
   by 1/P(included).

## The simulation

`simulation.py` implements a self-contained version of this pipeline:

1. **Generate a population** of 10,000 individuals with features (age,
   shelter status, service use, network size, subgroup) and a health outcome.
2. **Assign true selection probabilities.** VBS favors sheltered, high-service
   users. RDS favors large-network, marginalized subgroups.
3. **Draw samples:** 800 via VBS, 200 via RDS.
4. **Train random forests** (sklearn) to predict cross-method probabilities.
5. **Combine** using the inclusion probability formula.
6. **Compare estimators** against the known population truth.

## Quickstart

```bash
uv venv && uv pip install -r requirements.txt
.venv/bin/python simulation.py
```

This prints a comparison table and saves `results.png`.

## Results

With the default seed, the combined weighted estimator has substantially
less bias than VBS alone:

| Estimator              | Bias   |
|------------------------|--------|
| VBS unweighted         | +5.98  |
| VBS inverse-prob weight| +4.84  |
| Naive combined         | +4.43  |
| RDS unweighted         | -1.74  |
| **Combined weighted**  | **+2.53** |

The VBS sample barely reaches farmworkers (74/800) and DV survivors (86/800),
so even inverse-probability weighting within VBS alone cannot fully correct
the coverage gap. RDS fills that gap, and the random forest bridges the two
probability frameworks.

## Tests

```bash
.venv/bin/python test_simulation.py
```
