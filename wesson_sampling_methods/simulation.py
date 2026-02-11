"""
Simulation of the Wesson et al. (2025) method for combining
venue-based sampling (VBS) and respondent-driven sampling (RDS)
using random forests to predict cross-method selection probabilities.

Reference:
    Wesson P, Graham-Squire D, Perry E, Assaf RD, Kushel M.
    "Novel methods to construct a representative sample for surveying
    California's unhoused population: the California Statewide Study of
    People Experiencing Homelessness (CASPEH)."
    American Journal of Epidemiology. 2025;194(5):1238-1248.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC12055459/
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_population(n=10_000, seed=42):
    """Generate a synthetic hidden population with known characteristics.

    Each person has features that influence their probability of being
    reached by venue-based sampling (VBS) vs respondent-driven sampling (RDS).

    Returns a DataFrame with columns:
        age:            continuous, 18-70
        sheltered:      binary, whether the person sleeps in a shelter
        service_use:    continuous 0-1, how often they use services
        network_size:   integer 1-20, social network connections
        subgroup:       categorical, population subgroup (0-3)
        health_score:   continuous 0-100, outcome of interest
        p_vbs:          true VBS selection probability
        p_rds:          true RDS selection probability
    """
    rng = np.random.default_rng(seed)

    age = rng.uniform(18, 70, n)
    sheltered = rng.binomial(1, 0.4, n)
    service_use = rng.beta(2, 5, n)
    network_size = rng.poisson(5, n) + 1
    network_size = np.clip(network_size, 1, 20)
    # Subgroups: 0=general, 1=youth, 2=farmworker, 3=DV survivor
    subgroup = rng.choice([0, 1, 2, 3], n, p=[0.5, 0.2, 0.15, 0.15])

    # --- True selection probabilities (the "ground truth" we simulate) ---

    # VBS: higher for sheltered, high service users, older people
    # Lower for certain subgroups (farmworkers, DV survivors) who avoid venues
    logit_vbs = (
        -2.0
        + 1.5 * sheltered
        + 2.0 * service_use
        + 0.01 * age
        - 1.0 * (subgroup == 2).astype(float)  # farmworkers hard to reach at venues
        - 0.8 * (subgroup == 3).astype(float)  # DV survivors hard to reach at venues
    )
    p_vbs = 1 / (1 + np.exp(-logit_vbs))

    # RDS: higher for people with large networks, youth, marginalized subgroups
    # Lower for sheltered (less peer-network recruitment in shelters)
    logit_rds = (
        -3.0
        + 0.15 * network_size
        - 0.5 * sheltered
        + 1.2 * (subgroup == 1).astype(float)  # youth recruited via peers
        + 1.0 * (subgroup == 2).astype(float)  # farmworkers recruited via peers
        + 0.8 * (subgroup == 3).astype(float)  # DV survivors recruited via peers
    )
    p_rds = 1 / (1 + np.exp(-logit_rds))

    # Health outcome: differs by subgroup and age (this is what we want to estimate)
    health_score = (
        50
        + 5 * sheltered
        - 0.3 * age
        + 10 * service_use
        - 8 * (subgroup == 2).astype(float)
        - 5 * (subgroup == 3).astype(float)
        + rng.normal(0, 10, n)
    )
    health_score = np.clip(health_score, 0, 100)

    return pd.DataFrame(
        {
            "age": age,
            "sheltered": sheltered,
            "service_use": service_use,
            "network_size": network_size,
            "subgroup": subgroup,
            "health_score": health_score,
            "p_vbs": p_vbs,
            "p_rds": p_rds,
        }
    )


def draw_samples(pop, n_vbs=800, n_rds=200, seed=42):
    """Draw VBS and RDS samples from the population using true selection probs."""
    rng = np.random.default_rng(seed)

    vbs_selected = rng.random(len(pop)) < pop["p_vbs"]
    rds_selected = rng.random(len(pop)) < pop["p_rds"]

    vbs_pool = pop[vbs_selected].copy()
    rds_pool = pop[rds_selected].copy()

    # Subsample to target sizes
    if len(vbs_pool) > n_vbs:
        vbs_sample = vbs_pool.sample(n=n_vbs, random_state=seed)
    else:
        vbs_sample = vbs_pool

    if len(rds_pool) > n_rds:
        rds_sample = rds_pool.sample(n=n_rds, random_state=seed)
    else:
        rds_sample = rds_pool

    vbs_sample = vbs_sample.copy()
    rds_sample = rds_sample.copy()
    vbs_sample["method"] = "VBS"
    rds_sample["method"] = "RDS"

    return vbs_sample, rds_sample


def predict_cross_probabilities(pop, vbs_sample, rds_sample, seed=42):
    """Use random forests to predict each method's selection probability
    for participants sampled by the other method.

    This is the key methodological step from Wesson et al.:
    - Train RF on VBS sample to predict P(VBS selection) from features
    - Train RF on RDS sample to predict P(RDS selection) from features
    - Predict cross-probabilities for all sampled participants
    """
    features = ["age", "sheltered", "service_use", "network_size", "subgroup"]

    # --- RF model for VBS selection probability ---
    # Train on the full population indicator: who was in the VBS sample?
    # In practice, this is trained on VBS participants with known design-based
    # selection probabilities. Here we use the binary indicator as a simpler
    # analog.
    all_indices = set(pop.index)
    vbs_indices = set(vbs_sample.index)
    rds_indices = set(rds_sample.index)

    # For VBS model: label VBS participants as 1, a random subset of
    # non-VBS as 0 (case-control style, as in practice)
    non_vbs = pop.loc[list(all_indices - vbs_indices)]
    if len(non_vbs) > len(vbs_sample) * 3:
        non_vbs = non_vbs.sample(n=len(vbs_sample) * 3, random_state=seed)
    train_vbs = pd.concat(
        [
            vbs_sample.assign(label=1),
            non_vbs.assign(label=0),
        ]
    )

    rf_vbs = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
    rf_vbs.fit(train_vbs[features], train_vbs["label"])

    # For RDS model: label RDS participants as 1, non-RDS as 0
    non_rds = pop.loc[list(all_indices - rds_indices)]
    if len(non_rds) > len(rds_sample) * 3:
        non_rds = non_rds.sample(n=len(rds_sample) * 3, random_state=seed)
    train_rds = pd.concat(
        [
            rds_sample.assign(label=1),
            non_rds.assign(label=0),
        ]
    )

    rf_rds = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=seed)
    rf_rds.fit(train_rds[features], train_rds["label"])

    # Predict cross-probabilities
    vbs_sample = vbs_sample.copy()
    rds_sample = rds_sample.copy()

    # For VBS participants: predict their RDS selection probability
    vbs_sample["pred_p_rds"] = rf_rds.predict_proba(vbs_sample[features])[:, 1]
    # Their VBS probability is known from the design (use true p_vbs)
    vbs_sample["pred_p_vbs"] = vbs_sample["p_vbs"]

    # For RDS participants: predict their VBS selection probability
    rds_sample["pred_p_vbs"] = rf_vbs.predict_proba(rds_sample[features])[:, 1]
    # Their RDS probability is known from the design (use true p_rds)
    rds_sample["pred_p_rds"] = rds_sample["p_rds"]

    # AUC diagnostics
    auc_vbs = roc_auc_score(train_vbs["label"], rf_vbs.predict_proba(train_vbs[features])[:, 1])
    auc_rds = roc_auc_score(train_rds["label"], rf_rds.predict_proba(train_rds[features])[:, 1])

    # Variable importance
    imp_vbs = dict(zip(features, rf_vbs.feature_importances_))
    imp_rds = dict(zip(features, rf_rds.feature_importances_))

    return vbs_sample, rds_sample, {"auc_vbs": auc_vbs, "auc_rds": auc_rds, "imp_vbs": imp_vbs, "imp_rds": imp_rds}


def combine_samples(vbs_sample, rds_sample):
    """Combine VBS and RDS samples using the inclusion probability formula:

        P(overall) = P(VBS) + P(RDS) - P(VBS) * P(RDS)

    Then weight each participant by 1/P(overall).
    """
    combined = pd.concat([vbs_sample, rds_sample], ignore_index=True)

    # Overall inclusion probability (union of two independent events)
    combined["p_overall"] = (
        combined["pred_p_vbs"] + combined["pred_p_rds"] - combined["pred_p_vbs"] * combined["pred_p_rds"]
    )

    # Inverse probability weight
    combined["weight"] = 1.0 / combined["p_overall"]

    # Normalize weights to sum to population-like total
    combined["weight_normalized"] = combined["weight"] / combined["weight"].sum()

    return combined


def estimate_means(pop, vbs_sample, rds_sample, combined):
    """Compare population truth to various sample estimates."""
    true_mean = pop["health_score"].mean()
    true_subgroup = pop.groupby("subgroup")["health_score"].mean()

    # Unweighted estimates
    vbs_unweighted = vbs_sample["health_score"].mean()
    rds_unweighted = rds_sample["health_score"].mean()
    naive_combined = pd.concat([vbs_sample, rds_sample])["health_score"].mean()

    # VBS-only with inverse probability weights
    vbs_ipw = np.average(vbs_sample["health_score"], weights=1.0 / vbs_sample["p_vbs"])

    # Combined weighted estimate (the Wesson method)
    combined_weighted = np.average(combined["health_score"], weights=combined["weight"])

    # Subgroup estimates
    combined_subgroup = combined.groupby("subgroup").apply(
        lambda g: np.average(g["health_score"], weights=g["weight"]), include_groups=False
    )

    return {
        "true_mean": true_mean,
        "true_subgroup": true_subgroup,
        "vbs_unweighted": vbs_unweighted,
        "rds_unweighted": rds_unweighted,
        "naive_combined": naive_combined,
        "vbs_ipw": vbs_ipw,
        "combined_weighted": combined_weighted,
        "combined_subgroup": combined_subgroup,
    }


def plot_results(estimates, diagnostics, combined, pop, filename="results.png"):
    """Create a multi-panel figure summarizing the simulation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Panel 1: Overall mean estimates comparison ---
    ax = axes[0, 0]
    methods = ["VBS\nunweighted", "RDS\nunweighted", "Naive\ncombined", "VBS\nIPW", "Combined\nweighted\n(Wesson)"]
    values = [
        estimates["vbs_unweighted"],
        estimates["rds_unweighted"],
        estimates["naive_combined"],
        estimates["vbs_ipw"],
        estimates["combined_weighted"],
    ]
    colors = ["#4878CF", "#D65F5F", "#B47CC7", "#78B7C5", "#2CA02C"]
    bars = ax.bar(methods, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(estimates["true_mean"], color="black", linestyle="--", linewidth=2, label="True population mean")
    ax.set_ylabel("Mean health score")
    ax.set_title("Overall mean: estimator comparison")
    ax.legend()

    # Add bias annotations
    for bar, val in zip(bars, values):
        bias = val - estimates["true_mean"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"bias={bias:+.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # --- Panel 2: Subgroup estimates ---
    ax = axes[0, 1]
    subgroup_names = ["General", "Youth", "Farmworker", "DV survivor"]
    x = np.arange(len(subgroup_names))
    width = 0.35
    true_vals = [estimates["true_subgroup"].get(i, 0) for i in range(4)]
    est_vals = [estimates["combined_subgroup"].get(i, 0) for i in range(4)]
    ax.bar(x - width / 2, true_vals, width, label="True", color="#4878CF", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, est_vals, width, label="Wesson estimate", color="#2CA02C", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(subgroup_names)
    ax.set_ylabel("Mean health score")
    ax.set_title("Subgroup estimates: true vs combined weighted")
    ax.legend()

    # --- Panel 3: RF variable importance ---
    ax = axes[1, 0]
    features = list(diagnostics["imp_vbs"].keys())
    x = np.arange(len(features))
    width = 0.35
    ax.barh(
        x - width / 2,
        [diagnostics["imp_vbs"][f] for f in features],
        width,
        label=f'VBS model (AUC={diagnostics["auc_vbs"]:.2f})',
        color="#4878CF",
    )
    ax.barh(
        x + width / 2,
        [diagnostics["imp_rds"][f] for f in features],
        width,
        label=f'RDS model (AUC={diagnostics["auc_rds"]:.2f})',
        color="#D65F5F",
    )
    ax.set_yticks(x)
    ax.set_yticklabels(features)
    ax.set_xlabel("Feature importance")
    ax.set_title("Random forest variable importance")
    ax.legend(fontsize=8)

    # --- Panel 4: Selection probability distributions ---
    ax = axes[1, 1]
    vbs_in_combined = combined[combined["method"] == "VBS"]
    rds_in_combined = combined[combined["method"] == "RDS"]
    ax.hist(
        vbs_in_combined["pred_p_rds"],
        bins=30,
        alpha=0.6,
        color="#4878CF",
        label="VBS participants:\npredicted P(RDS)",
        density=True,
    )
    ax.hist(
        rds_in_combined["pred_p_rds"],
        bins=30,
        alpha=0.6,
        color="#D65F5F",
        label="RDS participants:\nknown P(RDS)",
        density=True,
    )
    ax.set_xlabel("P(RDS selection)")
    ax.set_ylabel("Density")
    ax.set_title("RDS selection probability distributions")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Wesson et al. (2025): Combining VBS + RDS via Random Forests",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {filename}")
    return fig


def run_simulation(seed=42):
    """Run the full simulation pipeline."""
    print("=" * 65)
    print("Wesson et al. (2025) Sampling Method Simulation")
    print("=" * 65)

    # Step 1: Generate population
    print("\n1. Generating synthetic population (N=10,000)...")
    pop = generate_population(seed=seed)
    print(f"   Population mean health score: {pop['health_score'].mean():.2f}")
    print(f"   Subgroup distribution: {pop['subgroup'].value_counts().sort_index().to_dict()}")
    print(f"   Mean P(VBS): {pop['p_vbs'].mean():.3f}")
    print(f"   Mean P(RDS): {pop['p_rds'].mean():.3f}")

    # Step 2: Draw samples
    print("\n2. Drawing VBS and RDS samples...")
    vbs, rds = draw_samples(pop, n_vbs=800, n_rds=200, seed=seed)
    print(f"   VBS sample: n={len(vbs)}")
    print(f"   RDS sample: n={len(rds)}")
    print(f"   VBS subgroup distribution: {vbs['subgroup'].value_counts().sort_index().to_dict()}")
    print(f"   RDS subgroup distribution: {rds['subgroup'].value_counts().sort_index().to_dict()}")

    # Step 3: Predict cross-probabilities with random forests
    print("\n3. Training random forests for cross-probability prediction...")
    vbs, rds, diagnostics = predict_cross_probabilities(pop, vbs, rds, seed=seed)
    print(f"   VBS model AUC: {diagnostics['auc_vbs']:.3f}")
    print(f"   RDS model AUC: {diagnostics['auc_rds']:.3f}")
    print(f"   VBS model top features: {sorted(diagnostics['imp_vbs'].items(), key=lambda x: -x[1])[:3]}")
    print(f"   RDS model top features: {sorted(diagnostics['imp_rds'].items(), key=lambda x: -x[1])[:3]}")

    # Step 4: Combine samples
    print("\n4. Combining samples with P(overall) = P(VBS) + P(RDS) - P(VBS)*P(RDS)...")
    combined = combine_samples(vbs, rds)
    print(f"   Combined sample: n={len(combined)}")
    print(f"   Mean overall inclusion probability: {combined['p_overall'].mean():.3f}")

    # Step 5: Compare estimates
    print("\n5. Comparing estimates to population truth...")
    estimates = estimate_means(pop, vbs, rds, combined)

    true = estimates["true_mean"]
    print(f"\n   {'Estimator':<30} {'Estimate':>10} {'Bias':>10}")
    print(f"   {'-'*50}")
    print(f"   {'True population mean':<30} {true:>10.2f} {0:>10.2f}")
    print(
        f"   {'VBS unweighted':<30} {estimates['vbs_unweighted']:>10.2f} {estimates['vbs_unweighted']-true:>10.2f}"
    )
    print(
        f"   {'RDS unweighted':<30} {estimates['rds_unweighted']:>10.2f} {estimates['rds_unweighted']-true:>10.2f}"
    )
    print(
        f"   {'Naive combined (unweighted)':<30} {estimates['naive_combined']:>10.2f} {estimates['naive_combined']-true:>10.2f}"
    )
    print(f"   {'VBS IPW only':<30} {estimates['vbs_ipw']:>10.2f} {estimates['vbs_ipw']-true:>10.2f}")
    print(
        f"   {'Combined weighted (Wesson)':<30} {estimates['combined_weighted']:>10.2f} {estimates['combined_weighted']-true:>10.2f}"
    )

    # Subgroup comparison
    print(f"\n   Subgroup estimates:")
    print(f"   {'Subgroup':<20} {'True':>10} {'Wesson':>10} {'Bias':>10}")
    print(f"   {'-'*50}")
    for i, name in enumerate(["General", "Youth", "Farmworker", "DV survivor"]):
        t = estimates["true_subgroup"].get(i, 0)
        e = estimates["combined_subgroup"].get(i, 0)
        print(f"   {name:<20} {t:>10.2f} {e:>10.2f} {e-t:>10.2f}")

    # Step 6: Plot
    print("\n6. Generating results figure...")
    plot_results(estimates, diagnostics, combined, pop)

    return pop, vbs, rds, combined, estimates, diagnostics


if __name__ == "__main__":
    run_simulation()
