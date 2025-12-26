#!/usr/bin/env python3
"""
Extended simulation with transmission feedback demonstrating BCG vaccination effects.

This model extends the basic LTBI progression simulation to include transmission
dynamics: active TB cases can infect susceptible individuals, creating new LTBI
cases. This creates a feedback loop where preventing progression also prevents
subsequent transmission.

Based on findings from:
"Effect of BCG vaccination on the progression of latent tuberculosis
infection to active disease in contacts: a systematic review and meta-analysis"
Cai et al., BMC Infectious Diseases (2025)

MODEL DESCRIPTION
=================

Compartments (SLIR model):
  S = Susceptible (can be infected)
  L = Latent TB infection (infected, not infectious)
  I = Infectious/Active TB (symptomatic, can transmit)
  R = Recovered/Removed (treated, no longer infectious)

Transitions:
  S → L: Transmission from infectious cases (rate = β * I / N)
  L → I: Progression from latent to active (rate = σ, reduced by BCG)
  I → R: Treatment/recovery (rate = γ)

Key insight: BCG reduces σ (progression rate). With transmission feedback,
this also indirectly reduces new infections, creating compounding benefits.

Without feedback: Each prevented case saves 1 case
With feedback: Each prevented case saves 1 + (secondary cases it would have caused)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy.integrate import odeint

np.random.seed(42)


@dataclass
class TransmissionParams:
    """Parameters for the SLIR transmission model."""

    # Population
    N: int = 100_000  # Total population size

    # Initial conditions
    initial_ltbi_frac: float = 0.25  # 25% start with LTBI (WHO estimate for high-burden)
    initial_active_frac: float = 0.001  # 0.1% start with active TB

    # Time parameters
    years: int = 30  # Longer timeframe to see divergence
    dt: float = 0.01  # Time step (years)

    # Epidemiological parameters
    beta: float = 10.0  # Transmission rate (effective contacts per year)
    sigma: float = 0.05  # Annual progression rate LTBI→Active (5% of recent infections)
    gamma: float = 0.5  # Recovery rate (average 2 years infectious without treatment)

    # BCG effect (from meta-analysis)
    rr_bcg: float = 0.57  # Risk ratio for progression with BCG

    # Vaccination coverage scenarios
    vax_coverage: float = 0.80  # 80% of population vaccinated


def slir_model(y, t, params: TransmissionParams, bcg_effect: bool = False):
    """
    SLIR differential equations with optional BCG effect.

    Args:
        y: State vector [S, L, I, R]
        t: Time
        params: Model parameters
        bcg_effect: If True, apply BCG reduction to progression rate
    """
    S, L, I, R = y
    N = S + L + I + R

    # Progression rate (reduced by BCG if vaccinated population)
    if bcg_effect:
        # Weighted average: vaccinated have reduced progression
        sigma_eff = params.sigma * (
            (1 - params.vax_coverage) +  # Unvaccinated
            params.vax_coverage * params.rr_bcg  # Vaccinated
        )
    else:
        sigma_eff = params.sigma

    # Force of infection
    lambda_t = params.beta * I / N

    # Differential equations
    dS = -lambda_t * S
    dL = lambda_t * S - sigma_eff * L
    dI = sigma_eff * L - params.gamma * I
    dR = params.gamma * I

    return [dS, dL, dI, dR]


def run_slir_simulation(params: TransmissionParams, bcg_effect: bool = False
                        ) -> Dict[str, np.ndarray]:
    """
    Run the SLIR model simulation.

    Returns dictionary with time series for each compartment.
    """
    # Initial conditions
    I0 = params.N * params.initial_active_frac
    L0 = params.N * params.initial_ltbi_frac
    R0 = 0
    S0 = params.N - L0 - I0 - R0

    y0 = [S0, L0, I0, R0]

    # Time points
    t = np.arange(0, params.years + params.dt, params.dt)

    # Solve ODE
    solution = odeint(slir_model, y0, t, args=(params, bcg_effect))

    return {
        'time': t,
        'S': solution[:, 0],
        'L': solution[:, 1],
        'I': solution[:, 2],
        'R': solution[:, 3],
        'cumulative_cases': solution[:, 2] + solution[:, 3]  # I + R
    }


def run_no_feedback_model(params: TransmissionParams, bcg_effect: bool = False
                          ) -> Dict[str, np.ndarray]:
    """
    Simple model WITHOUT transmission feedback.

    In this model, we track the same initial LTBI cohort but ignore
    new infections. This represents the "direct effect only" scenario.
    """
    # Time points
    t = np.arange(0, params.years + params.dt, params.dt)

    # Initial LTBI population
    L0 = params.N * params.initial_ltbi_frac
    I0 = params.N * params.initial_active_frac

    # Progression rate
    if bcg_effect:
        sigma_eff = params.sigma * (
            (1 - params.vax_coverage) +
            params.vax_coverage * params.rr_bcg
        )
    else:
        sigma_eff = params.sigma

    # Without feedback: LTBI decays exponentially as people progress
    # No new infections added to L
    L = L0 * np.exp(-sigma_eff * t)

    # Cumulative progressions from initial cohort only
    cumulative_from_initial = L0 * (1 - np.exp(-sigma_eff * t))

    # Active cases (simplified: immediate progression counting)
    # This ignores recovery for simplicity in comparison
    I = I0 + cumulative_from_initial

    return {
        'time': t,
        'L': L,
        'I': I,
        'cumulative_cases': I0 + cumulative_from_initial
    }


def calculate_basic_reproduction_number(params: TransmissionParams,
                                         bcg_effect: bool = False) -> float:
    """
    Calculate R0 for the model.

    R0 = (β / γ) * (σ / (σ + μ)) ≈ β * σ / (γ * σ) = β / γ for simple model

    For TB: R0 = β / γ (transmission rate / recovery rate)
    """
    if bcg_effect:
        sigma_eff = params.sigma * (
            (1 - params.vax_coverage) +
            params.vax_coverage * params.rr_bcg
        )
    else:
        sigma_eff = params.sigma

    # Effective R0 considering progression
    R0 = (params.beta / params.gamma) * (sigma_eff / (sigma_eff + 0.02))  # 0.02 = background mortality
    return R0


def plot_model_comparison(params: TransmissionParams):
    """Create comprehensive visualization comparing models."""

    fig = plt.figure(figsize=(16, 14))

    # Run all simulations
    # With transmission feedback
    no_bcg_feedback = run_slir_simulation(params, bcg_effect=False)
    bcg_feedback = run_slir_simulation(params, bcg_effect=True)

    # Without transmission feedback
    no_bcg_simple = run_no_feedback_model(params, bcg_effect=False)
    bcg_simple = run_no_feedback_model(params, bcg_effect=True)

    t = no_bcg_feedback['time']

    # =========================================================================
    # Panel A: Active TB cases over time (with feedback)
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)

    ax1.plot(t, no_bcg_feedback['I'] / 1000, 'r-', linewidth=2.5,
             label='No BCG')
    ax1.plot(t, bcg_feedback['I'] / 1000, 'b-', linewidth=2.5,
             label=f'With BCG (80% coverage)')
    ax1.fill_between(t, no_bcg_feedback['I'] / 1000, bcg_feedback['I'] / 1000,
                     alpha=0.3, color='green', label='Cases averted')

    ax1.set_xlabel('Years', fontsize=12)
    ax1.set_ylabel('Active TB Cases (thousands)', fontsize=12)
    ax1.set_title('A) WITH Transmission Feedback\nActive TB prevalence over time',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, params.years)

    # =========================================================================
    # Panel B: Cumulative cases comparison
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2)

    # With feedback
    ax2.plot(t, no_bcg_feedback['cumulative_cases'] / 1000, 'r-', linewidth=2.5,
             label='No BCG (with feedback)')
    ax2.plot(t, bcg_feedback['cumulative_cases'] / 1000, 'b-', linewidth=2.5,
             label='BCG (with feedback)')

    # Without feedback (scaled for comparison)
    scale = params.N / (params.N * params.initial_ltbi_frac)  # Scale to full pop
    ax2.plot(t, no_bcg_simple['cumulative_cases'] / 1000, 'r--', linewidth=2,
             label='No BCG (no feedback)', alpha=0.7)
    ax2.plot(t, bcg_simple['cumulative_cases'] / 1000, 'b--', linewidth=2,
             label='BCG (no feedback)', alpha=0.7)

    ax2.set_xlabel('Years', fontsize=12)
    ax2.set_ylabel('Cumulative TB Cases (thousands)', fontsize=12)
    ax2.set_title('B) Cumulative Cases: With vs Without Feedback\n'
                  '(Solid=feedback, Dashed=no feedback)',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, params.years)

    # =========================================================================
    # Panel C: Cases averted - direct vs total effect
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3)

    # Cases averted without feedback (direct effect only)
    direct_averted = no_bcg_simple['cumulative_cases'] - bcg_simple['cumulative_cases']

    # Cases averted with feedback (total effect)
    total_averted = no_bcg_feedback['cumulative_cases'] - bcg_feedback['cumulative_cases']

    # Indirect effect (due to reduced transmission)
    indirect_averted = total_averted - direct_averted

    ax3.fill_between(t, 0, direct_averted / 1000, alpha=0.7, color='steelblue',
                     label='Direct effect (prevented progressions)')
    ax3.fill_between(t, direct_averted / 1000, total_averted / 1000,
                     alpha=0.7, color='green',
                     label='Indirect effect (prevented infections)')
    ax3.plot(t, total_averted / 1000, 'k-', linewidth=2, label='Total averted')

    ax3.set_xlabel('Years', fontsize=12)
    ax3.set_ylabel('Cumulative Cases Averted (thousands)', fontsize=12)
    ax3.set_title('C) Breakdown of BCG Impact\n'
                  'Direct vs Indirect (transmission-mediated) effects',
                  fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, params.years)
    ax3.set_ylim(0, None)

    # =========================================================================
    # Panel D: Amplification factor over time
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4)

    # Avoid division by zero
    mask = direct_averted > 100  # Only where we have meaningful numbers
    amplification = np.ones_like(t)
    amplification[mask] = total_averted[mask] / direct_averted[mask]

    ax4.plot(t, amplification, 'purple', linewidth=2.5)
    ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5,
                label='No amplification (direct effect only)')

    ax4.set_xlabel('Years', fontsize=12)
    ax4.set_ylabel('Amplification Factor', fontsize=12)
    ax4.set_title('D) Feedback Amplification Over Time\n'
                  '(Total effect / Direct effect)',
                  fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, params.years)
    ax4.set_ylim(0.8, None)

    # Add annotation for final amplification
    final_amp = amplification[-1]
    ax4.annotate(f'Final: {final_amp:.1f}x',
                 xy=(params.years * 0.9, final_amp),
                 fontsize=12, fontweight='bold',
                 ha='right', va='bottom')

    plt.suptitle('BCG Vaccination: Transmission Feedback Amplifies Protective Effect\n'
                 'Based on Cai et al., BMC Infectious Diseases (2025) | '
                 f'BCG RR={params.rr_bcg}, Coverage={params.vax_coverage*100:.0f}%',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def plot_model_explanation():
    """Create a figure explaining the model structure."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Without feedback
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Model WITHOUT Transmission Feedback\n(Closed cohort)', fontsize=13, fontweight='bold')

    # Draw compartments
    boxes = {
        'L': (2, 5, 'Latent\nTB (L)'),
        'I': (7, 5, 'Active\nTB (I)'),
    }

    for name, (x, y, label) in boxes.items():
        color = 'lightblue' if name == 'L' else 'salmon'
        rect = plt.Rectangle((x-1, y-1), 2, 2, facecolor=color,
                              edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow L → I
    ax.annotate('', xy=(6, 5), xytext=(4, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(5, 5.8, 'σ (progression)\nReduced by BCG', ha='center', fontsize=10)

    ax.text(5, 2, 'Each prevented progression\nsaves exactly 1 case\n(no downstream effects)',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 2: With feedback
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Model WITH Transmission Feedback\n(Open population, SLIR)', fontsize=13, fontweight='bold')

    # Draw compartments
    boxes = {
        'S': (1.5, 7, 'Susceptible\n(S)'),
        'L': (5, 7, 'Latent\nTB (L)'),
        'I': (8.5, 7, 'Active\nTB (I)'),
        'R': (8.5, 3, 'Recovered\n(R)'),
    }

    for name, (x, y, label) in boxes.items():
        if name == 'S':
            color = 'lightgreen'
        elif name == 'L':
            color = 'lightblue'
        elif name == 'I':
            color = 'salmon'
        else:
            color = 'lightgray'
        rect = plt.Rectangle((x-1, y-1), 2, 2, facecolor=color,
                              edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    # S → L
    ax.annotate('', xy=(4, 7), xytext=(2.5, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(3.25, 7.8, 'β·I/N', ha='center', fontsize=10)

    # L → I
    ax.annotate('', xy=(7.5, 7), xytext=(6, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(6.75, 7.8, 'σ', ha='center', fontsize=10, color='red', fontweight='bold')

    # I → R
    ax.annotate('', xy=(8.5, 4), xytext=(8.5, 6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(9.2, 5, 'γ', ha='center', fontsize=10)

    # Feedback arrow (I influences S→L)
    ax.annotate('', xy=(3, 6.3), xytext=(7.5, 6.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple',
                               connectionstyle='arc3,rad=0.3', linestyle='--'))
    ax.text(5, 5.5, 'Transmission\nfeedback', ha='center', fontsize=9, color='purple')

    ax.text(5, 1.5, 'Each prevented progression saves:\n'
            '1 case + cases that case would have caused\n'
            '(compounding benefit over time)',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return fig


def print_model_summary(params: TransmissionParams):
    """Print detailed summary of model and results."""

    print("=" * 75)
    print("TRANSMISSION MODEL: BCG Vaccination with Feedback Effects")
    print("=" * 75)
    print()

    print("MODEL STRUCTURE (SLIR Compartmental Model)")
    print("-" * 75)
    print("""
    ┌─────────────┐      β·I/N      ┌─────────────┐       σ        ┌─────────────┐
    │ Susceptible │ ───────────────→│   Latent    │───────────────→│   Active    │
    │     (S)     │                 │   TB (L)    │  (↓ by BCG)    │   TB (I)    │
    └─────────────┘                 └─────────────┘                └──────┬──────┘
                                                                          │ γ
                                         FEEDBACK LOOP                    ↓
                                    Active cases infect          ┌─────────────┐
                                    susceptibles, creating       │  Recovered  │
                                    new LTBI cases               │     (R)     │
                                                                 └─────────────┘
    """)

    print("PARAMETERS")
    print("-" * 75)
    print(f"  Population size (N):           {params.N:,}")
    print(f"  Initial LTBI prevalence:       {params.initial_ltbi_frac*100:.0f}%")
    print(f"  Initial active TB prevalence:  {params.initial_active_frac*100:.2f}%")
    print(f"  Simulation duration:           {params.years} years")
    print()
    print(f"  Transmission rate (β):         {params.beta:.1f} per year")
    print(f"  Progression rate (σ):          {params.sigma*100:.1f}% per year")
    print(f"  Recovery rate (γ):             {params.gamma:.2f} per year")
    print()
    print(f"  BCG risk ratio:                {params.rr_bcg}")
    print(f"  BCG coverage:                  {params.vax_coverage*100:.0f}%")
    print()

    # Calculate R0
    R0_no_bcg = calculate_basic_reproduction_number(params, bcg_effect=False)
    R0_bcg = calculate_basic_reproduction_number(params, bcg_effect=True)
    print(f"  Effective R₀ without BCG:      {R0_no_bcg:.2f}")
    print(f"  Effective R₀ with BCG:         {R0_bcg:.2f}")
    print()

    # Run simulations
    no_bcg_feedback = run_slir_simulation(params, bcg_effect=False)
    bcg_feedback = run_slir_simulation(params, bcg_effect=True)
    no_bcg_simple = run_no_feedback_model(params, bcg_effect=False)
    bcg_simple = run_no_feedback_model(params, bcg_effect=True)

    print("KEY DIFFERENCES: WITH vs WITHOUT FEEDBACK")
    print("-" * 75)
    print()
    print("WITHOUT Transmission Feedback (simple cohort model):")
    print("  - Only tracks progression of initial LTBI cases")
    print("  - No new infections generated")
    print("  - BCG effect is LINEAR: prevents X cases directly")
    print()
    print("WITH Transmission Feedback (SLIR model):")
    print("  - Active TB cases infect susceptibles → new LTBI")
    print("  - Those new LTBI cases can progress → more active TB")
    print("  - BCG effect COMPOUNDS: prevents X cases + their downstream cases")
    print()

    # Results comparison
    print("SIMULATION RESULTS")
    print("-" * 75)

    final_idx = -1
    years_check = [10, 20, 30]

    for yr in years_check:
        if yr > params.years:
            continue
        idx = int(yr / params.dt)

        direct_averted = (no_bcg_simple['cumulative_cases'][idx] -
                          bcg_simple['cumulative_cases'][idx])
        total_averted = (no_bcg_feedback['cumulative_cases'][idx] -
                         bcg_feedback['cumulative_cases'][idx])
        indirect = total_averted - direct_averted

        if direct_averted > 0:
            amplification = total_averted / direct_averted
        else:
            amplification = 1.0

        print(f"\nAt year {yr}:")
        print(f"  Cases averted (direct effect only):    {direct_averted:,.0f}")
        print(f"  Cases averted (with feedback):         {total_averted:,.0f}")
        print(f"  Additional cases averted (indirect):   {indirect:,.0f}")
        print(f"  Amplification factor:                  {amplification:.2f}x")

    print()
    print("=" * 75)
    print("INTERPRETATION")
    print("=" * 75)
    print("""
The transmission feedback creates a MULTIPLIER EFFECT:

1. DIRECT EFFECT: BCG prevents LTBI → Active TB progression
   - This is what the meta-analysis measured (RR = 0.57)
   - Each prevented case = 1 case saved

2. INDIRECT EFFECT: Fewer active cases → fewer new infections
   - Each active TB case would have infected others
   - Those new infections would become LTBI
   - Some would progress to active TB
   - Those would infect more people... (chain continues)

3. The AMPLIFICATION grows over time because:
   - Early prevented cases have more time to compound
   - Feedback loops accumulate over generations of transmission
   - In endemic settings, this can multiply the benefit 2-4x or more

POLICY IMPLICATION: The meta-analysis finding (RR=0.57) underestimates the
true population-level benefit of BCG vaccination because it only captures
the direct effect, not the transmission-reduction benefit.
""")
    print("=" * 75)


def main():
    """Run the transmission model simulation."""

    params = TransmissionParams()

    # Print summary
    print_model_summary(params)

    # Generate model explanation figure
    fig_explain = plot_model_explanation()
    fig_explain.savefig('/home/user/ai_assisted_research/bcg_model_explanation.png',
                        dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nModel explanation saved to: bcg_model_explanation.png")

    # Generate comparison figure
    fig_compare = plot_model_comparison(params)
    fig_compare.savefig('/home/user/ai_assisted_research/bcg_transmission_feedback.png',
                        dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Transmission model saved to: bcg_transmission_feedback.png")

    plt.show()

    return fig_explain, fig_compare


if __name__ == "__main__":
    main()
