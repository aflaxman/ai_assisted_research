"""
Anthrax Outbreak Simulation with Environmental Transmission.

This script simulates an anthrax outbreak in an animal population where:
1. Animals get infected from environmental spores
2. Infected animals die and become carcasses
3. Carcasses decompose and release more spores (feedback loop)
4. Scavengers can remove carcasses before they release spores

The simulation compares different scavenging rates to show how scavenging
can interrupt the disease transmission cycle.
"""

import numpy as np
import matplotlib.pyplot as plt
from laser.core import PropertySet
from laser.core.distributions import poisson
from laser.generic.model import Model
from laser.generic.components import Susceptible
from laser.core.utils import grid
from components import (
    EnvironmentalTransmission,
    InfectionToCarcass,
    CarcassDynamics,
    SporeEnvironment,
)
from utils import save_and_close_plot

def run_simulation(scavenging_rate=0.0):
    """
    Run an anthrax outbreak simulation.

    This function sets up and runs a complete simulation of environmental
    anthrax transmission on a spatial grid.

    Args:
        scavenging_rate: Probability that a carcass is scavenged each day (0.0 to 1.0)
                        Higher values = more scavenging = fewer spores = smaller outbreak

    Returns:
        model: Completed LASER model with all simulation results
    """
    # ==============================================================================
    # STEP 1: Define simulation parameters
    # ==============================================================================
    params = PropertySet({
        "nticks": 200,  # Simulation length in days
        "beta": 0.005,  # Infection rate per spore (higher = more contagious)
        "mean_infectious_period": 3.0,  # Average days until infected animal dies
        "mean_decomposition_period": 7.0,  # Average days for carcass to decompose
        "initial_spores_at_hotspot": 1000.0,  # Initial contamination level
        "spores_per_carcass": 100.0,  # Spores released per decomposed carcass
        "seed": 42,  # Random seed for reproducibility
    })

    # ==============================================================================
    # STEP 2: Create spatial scenario (10x10 grid, 100 animals per location)
    # ==============================================================================
    M, N = 10, 10
    scenario = grid(M=M, N=N, population_fn=lambda r, c: 100)

    # Initialize all animals as susceptible
    scenario["S"] = scenario["population"]
    # Note: I, C, and R counts are initialized as vector properties by the
    # components to track time-series, so we don't set them here.

    # ==============================================================================
    # STEP 3: Build model and configure timing distributions
    # ==============================================================================
    model = Model(scenario, params)

    # Use Poisson distributions for timing variability
    infdist = poisson(params.mean_infectious_period)
    decompdist = poisson(params.mean_decomposition_period)

    # ==============================================================================
    # STEP 4: Attach disease components
    #
    # IMPORTANT: Order matters! Components execute in this sequence each day:
    #   1. Susceptible - Tracks susceptible population
    #   2. EnvironmentalTransmission - Animals get infected from spores (S → I)
    #   3. InfectionToCarcass - Infected animals die (I → Carcass)
    #   4. CarcassDynamics - Carcasses decompose or get scavenged (C → Removed)
    #   5. SporeEnvironment - Updates environmental spore counts
    # ==============================================================================
    model.components = [
        Susceptible(model),
        EnvironmentalTransmission(model, beta=params.beta, infdurdist=infdist),
        InfectionToCarcass(model, decomp_dist=decompdist),
        CarcassDynamics(model, scavenging_rate=scavenging_rate,
                       spores_per_carcass=params.spores_per_carcass),
        SporeEnvironment(model, decay_rate=0.02)  # 2% daily spore decay
    ]

    # ==============================================================================
    # STEP 5: Seed the outbreak (place initial spores at center of grid)
    # ==============================================================================
    center_node = (M // 2) * N + (N // 2)
    model.nodes.spores[0, center_node] = params.initial_spores_at_hotspot

    # ==============================================================================
    # STEP 6: Run the simulation!
    # ==============================================================================
    model.run()

    return model

def plot_population_dynamics(model, title="Simulation Results"):
    """
    Plot infected and carcass counts over time.

    Shows the epidemic curve (infections) and carcass accumulation,
    which helps visualize the outbreak dynamics.

    Args:
        model: Completed simulation model
        title: Plot title (also used for filename)
    """
    # Extract total counts across all locations
    time = np.arange(model.params.nticks + 1)
    I = model.nodes.I.sum(axis=1)  # Total infected at each time
    C = model.nodes.C.sum(axis=1)  # Total carcasses at each time

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time, I, label="Infected", color='red', linewidth=2)
    ax1.plot(time, C, label="Carcass", color='black', linewidth=2)
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Population")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    plt.title(title)

    # Save using utility function
    save_and_close_plot(title)

def plot_spore_comparison(models_dict, title="Comparative Spore Dynamics"):
    """
    Compare environmental spore levels across different scenarios.

    This shows how different scavenging rates affect the spore feedback loop:
    - Low scavenging → many decomposing carcasses → high spore levels
    - High scavenging → few decomposing carcasses → low spore levels

    Args:
        models_dict: Dictionary mapping scenario names to completed models
        title: Plot title (also used for filename)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each scenario
    for name, model in models_dict.items():
        time = np.arange(model.params.nticks + 1)
        Spores = model.nodes.spores.sum(axis=1)
        ax.plot(time, Spores, label=name, linewidth=2)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Total Environmental Spores")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save using utility function
    save_and_close_plot(title)

def plot_spatial_comparison(models_dict, title="Spatial Spore Distribution (Final Day)"):
    """
    Create side-by-side heatmaps showing final spore distribution.

    This spatial view shows where spores accumulated on the grid,
    revealing how the outbreak spread from the initial hotspot.

    Args:
        models_dict: Dictionary mapping scenario names to completed models
        title: Plot title (also used for filename)
    """
    n_scenarios = len(models_dict)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5), sharey=True)

    # Grid dimensions (must match run_simulation)
    M, N = 10, 10

    # Handle single scenario case
    if n_scenarios == 1:
        axes = [axes]

    # Find global maximum for consistent color scale across all plots
    max_spores = 0
    for model in models_dict.values():
        max_spores = max(max_spores, model.nodes.spores[-1].max())

    # Create heatmap for each scenario
    for ax, (name, model) in zip(axes, models_dict.items()):
        # Get final day spore distribution
        final_spores = model.nodes.spores[-1]
        grid_spores = final_spores.reshape((M, N))

        # Plot heatmap
        im = ax.imshow(grid_spores, cmap='YlGn', vmin=0, vmax=max_spores,
                      interpolation='nearest')
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    # Add shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("Spore Concentration")

    plt.suptitle(title)

    # Save using utility function (skip tight_layout as it conflicts with suptitle)
    save_and_close_plot(title, tight_layout=False)


def main():
    """
    Run multiple simulations with different scavenging rates and compare results.

    This demonstrates how scavenging interrupts the disease transmission cycle:
    1. Baseline (0% scavenging): Maximum outbreak, all carcasses release spores
    2. Moderate (20% scavenging): Reduced outbreak
    3. High (50% scavenging): Significantly reduced outbreak

    The comparisons show the importance of carcass management in controlling
    environmental disease outbreaks like anthrax.
    """
    models = {}

    # Run three scenarios with different scavenging rates
    print("Running Baseline (Scavenging = 0.0)...")
    models["Baseline (0.0)"] = run_simulation(scavenging_rate=0.0)
    plot_population_dynamics(models["Baseline (0.0)"],
                            "Baseline Anthrax Outbreak (No Scavenging)")

    print("Running with Scavenging (Rate = 0.2)...")
    models["Scavenging (0.2)"] = run_simulation(scavenging_rate=0.2)
    plot_population_dynamics(models["Scavenging (0.2)"],
                            "Anthrax Outbreak with High Scavenging")

    print("Running with Scavenging (Rate = 0.5)...")
    models["Scavenging (0.5)"] = run_simulation(scavenging_rate=0.5)
    plot_population_dynamics(models["Scavenging (0.5)"],
                            "Anthrax Outbreak with Very High Scavenging")

    # Create comparative visualizations
    print("Generating comparative plots...")
    plot_spore_comparison(models)
    plot_spatial_comparison(models)

if __name__ == "__main__":
    main()
