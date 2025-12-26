#!/usr/bin/env python3
"""
Simulation demonstrating the main findings from:

"Effect of BCG vaccination on the progression of latent tuberculosis
infection to active disease in contacts: a systematic review and meta-analysis"

Cai et al., BMC Infectious Diseases (2025)
DOI: 10.1186/s12879-025-12318-y

Key findings from the meta-analysis:
- Overall: BCG vaccination associated with RR = 0.57 (95% CI: 0.40-0.82)
  for progression from LTBI to active TB
- Low TB incidence settings: RR = 0.48 (95% CI: 0.31-0.74)
- High TB incidence settings: RR = 0.71 (95% CI: 0.27-1.83) - not significant
- Children <15 years: RR = 0.44 (95% CI: 0.33-0.58)
- Adults ≥15 years: RR = 0.90 (95% CI: 0.31-2.64) - not significant
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

# Set random seed for reproducibility
np.random.seed(42)


@dataclass
class SimulationParams:
    """Parameters for LTBI progression simulation."""
    n_individuals: int = 10000  # Cohort size
    follow_up_years: int = 10   # Years of follow-up

    # Baseline annual progression rate from LTBI to active TB (unvaccinated)
    # Literature suggests ~5-10% lifetime risk, or roughly 0.5-1% per year
    baseline_annual_rate: float = 0.01  # 1% per year for unvaccinated

    # Risk ratios from the meta-analysis
    rr_overall: float = 0.57
    rr_low_incidence: float = 0.48
    rr_high_incidence: float = 0.71
    rr_children: float = 0.44
    rr_adults: float = 0.90

    # Confidence intervals (for uncertainty visualization)
    rr_overall_ci: Tuple[float, float] = (0.40, 0.82)
    rr_children_ci: Tuple[float, float] = (0.33, 0.58)
    rr_adults_ci: Tuple[float, float] = (0.31, 2.64)


def simulate_cohort(n: int, annual_rate: float, years: int) -> np.ndarray:
    """
    Simulate LTBI to active TB progression for a cohort.

    Returns array of shape (years+1,) with cumulative proportion progressed.
    """
    # Each year, individuals who haven't progressed have a chance to progress
    still_latent = np.ones(n, dtype=bool)
    cumulative_active = np.zeros(years + 1)

    for year in range(1, years + 1):
        # Determine who progresses this year
        progresses = np.random.random(n) < annual_rate
        new_cases = still_latent & progresses
        still_latent[new_cases] = False
        cumulative_active[year] = (~still_latent).sum() / n

    return cumulative_active


def run_comparison_simulation(params: SimulationParams,
                               rr: float,
                               label: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run simulation comparing vaccinated vs unvaccinated groups.
    """
    # Unvaccinated group
    unvax = simulate_cohort(
        params.n_individuals,
        params.baseline_annual_rate,
        params.follow_up_years
    )

    # Vaccinated group (reduced rate based on RR)
    vax_rate = params.baseline_annual_rate * rr
    vax = simulate_cohort(
        params.n_individuals,
        vax_rate,
        params.follow_up_years
    )

    return unvax, vax


def plot_main_findings(params: SimulationParams):
    """Create visualization of main simulation findings."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    years = np.arange(params.follow_up_years + 1)

    # Panel A: Overall effect
    ax = axes[0, 0]
    unvax, vax = run_comparison_simulation(params, params.rr_overall, "Overall")

    ax.plot(years, unvax * 100, 'r-', linewidth=2.5, label='Unvaccinated')
    ax.plot(years, vax * 100, 'b-', linewidth=2.5, label='BCG Vaccinated')
    ax.fill_between(years, unvax * 100, vax * 100, alpha=0.3, color='green',
                    label=f'Cases prevented (RR={params.rr_overall})')
    ax.set_xlabel('Years of Follow-up', fontsize=11)
    ax.set_ylabel('Cumulative % Progressed to Active TB', fontsize=11)
    ax.set_title('A) Overall Effect of BCG Vaccination\n(RR = 0.57, 95% CI: 0.40-0.82)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, params.follow_up_years)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    # Panel B: Low vs High incidence settings
    ax = axes[0, 1]
    _, vax_low = run_comparison_simulation(params, params.rr_low_incidence, "Low")
    _, vax_high = run_comparison_simulation(params, params.rr_high_incidence, "High")
    unvax, _ = run_comparison_simulation(params, 1.0, "Baseline")

    ax.plot(years, unvax * 100, 'r--', linewidth=2, label='Unvaccinated (baseline)')
    ax.plot(years, vax_low * 100, 'b-', linewidth=2.5,
            label=f'BCG in Low-incidence (RR={params.rr_low_incidence})')
    ax.plot(years, vax_high * 100, 'orange', linewidth=2.5, linestyle='-',
            label=f'BCG in High-incidence (RR={params.rr_high_incidence})')
    ax.set_xlabel('Years of Follow-up', fontsize=11)
    ax.set_ylabel('Cumulative % Progressed to Active TB', fontsize=11)
    ax.set_title('B) Effect by TB Incidence Setting\n(Significant only in low-incidence areas)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, params.follow_up_years)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    # Panel C: Children vs Adults
    ax = axes[1, 0]
    _, vax_children = run_comparison_simulation(params, params.rr_children, "Children")
    _, vax_adults = run_comparison_simulation(params, params.rr_adults, "Adults")

    ax.plot(years, unvax * 100, 'r--', linewidth=2, label='Unvaccinated (baseline)')
    ax.plot(years, vax_children * 100, 'green', linewidth=2.5,
            label=f'BCG in Children <15y (RR={params.rr_children})')
    ax.plot(years, vax_adults * 100, 'purple', linewidth=2.5,
            label=f'BCG in Adults ≥15y (RR={params.rr_adults})')
    ax.set_xlabel('Years of Follow-up', fontsize=11)
    ax.set_ylabel('Cumulative % Progressed to Active TB', fontsize=11)
    ax.set_title('C) Effect by Age Group\n(Significant only in children <15 years)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, params.follow_up_years)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    # Panel D: Summary bar chart of risk ratios
    ax = axes[1, 1]

    categories = ['Overall', 'Low\nIncidence', 'High\nIncidence', 'Children\n<15y', 'Adults\n≥15y']
    rrs = [params.rr_overall, params.rr_low_incidence, params.rr_high_incidence,
           params.rr_children, params.rr_adults]

    # CI bounds for error bars
    ci_lower = [params.rr_overall_ci[0], 0.31, 0.27,
                params.rr_children_ci[0], params.rr_adults_ci[0]]
    ci_upper = [params.rr_overall_ci[1], 0.74, 1.83,
                params.rr_children_ci[1], params.rr_adults_ci[1]]

    errors = [[rr - low for rr, low in zip(rrs, ci_lower)],
              [high - rr for rr, high in zip(rrs, ci_upper)]]

    colors = ['steelblue', 'green', 'orange', 'green', 'gray']

    bars = ax.bar(categories, rrs, color=colors, alpha=0.7, edgecolor='black')
    ax.errorbar(categories, rrs, yerr=errors, fmt='none', color='black',
                capsize=5, capthick=2, linewidth=2)

    # Reference line at RR = 1 (no effect)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
               label='No effect (RR=1)')

    ax.set_ylabel('Risk Ratio (95% CI)', fontsize=11)
    ax.set_title('D) Summary of Risk Ratios by Subgroup\n(Below red line = protective effect)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 3.0)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add significance annotations
    for i, (rr, upper) in enumerate(zip(rrs, ci_upper)):
        if upper < 1.0:  # Significant protective effect
            ax.annotate('*', (i, rr + errors[1][i] + 0.1),
                       ha='center', fontsize=16, fontweight='bold')

    ax.text(0.5, -0.15, '* Statistically significant (95% CI excludes 1.0)',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.suptitle('Simulation of BCG Vaccination Effect on LTBI → Active TB Progression\n'
                 'Based on Cai et al., BMC Infectious Diseases (2025)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def print_summary_statistics(params: SimulationParams):
    """Print summary of simulation results."""
    print("=" * 70)
    print("SIMULATION SUMMARY: BCG Vaccination and LTBI Progression")
    print("Based on: Cai et al., BMC Infectious Diseases (2025)")
    print("DOI: 10.1186/s12879-025-12318-y")
    print("=" * 70)
    print()

    print(f"Simulation Parameters:")
    print(f"  - Cohort size: {params.n_individuals:,} individuals with LTBI")
    print(f"  - Follow-up period: {params.follow_up_years} years")
    print(f"  - Baseline annual progression rate: {params.baseline_annual_rate*100:.1f}%")
    print()

    # Calculate expected outcomes
    print("Expected 10-year progression rates:")
    print("-" * 50)

    baseline_10yr = 1 - (1 - params.baseline_annual_rate) ** 10

    groups = [
        ("Unvaccinated (baseline)", 1.0),
        ("BCG Vaccinated (overall)", params.rr_overall),
        ("BCG in low-incidence setting", params.rr_low_incidence),
        ("BCG in high-incidence setting", params.rr_high_incidence),
        ("BCG in children <15y", params.rr_children),
        ("BCG in adults ≥15y", params.rr_adults),
    ]

    for name, rr in groups:
        rate = params.baseline_annual_rate * rr
        progression_10yr = 1 - (1 - rate) ** 10
        cases_per_1000 = progression_10yr * 1000
        print(f"  {name:35s}: {progression_10yr*100:5.2f}% ({cases_per_1000:.1f} per 1000)")

    print()
    print("Key Findings from the Meta-Analysis:")
    print("-" * 50)
    print("1. Overall: BCG vaccination reduces LTBI→active TB progression")
    print("   by 43% (RR=0.57, 95% CI: 0.40-0.82)")
    print()
    print("2. Effect is STRONGEST in:")
    print("   - Low TB incidence settings (52% reduction)")
    print("   - Children under 15 years (56% reduction)")
    print()
    print("3. Effect is NOT SIGNIFICANT in:")
    print("   - High TB incidence settings (95% CI crosses 1.0)")
    print("   - Adults 15 years and older (95% CI crosses 1.0)")
    print()
    print("Clinical Implication: BCG vaccination provides meaningful")
    print("protection against LTBI progression, particularly for children")
    print("in low TB burden settings.")
    print("=" * 70)


def main():
    """Run the simulation and generate outputs."""
    params = SimulationParams()

    # Print summary statistics
    print_summary_statistics(params)

    # Generate visualization
    fig = plot_main_findings(params)

    # Save figure
    output_path = '/home/user/ai_assisted_research/bcg_ltbi_simulation.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nFigure saved to: {output_path}")

    # Also display if running interactively
    plt.show()

    return fig


if __name__ == "__main__":
    main()
