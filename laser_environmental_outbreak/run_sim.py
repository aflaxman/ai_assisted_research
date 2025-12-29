
import numpy as np
import matplotlib.pyplot as plt
from laser.core import PropertySet
from laser.core.distributions import poisson
from laser.generic.model import Model
from laser.generic.components import Susceptible
from laser.core.utils import grid
from laser_environmental_outbreak.components import (
    EnvironmentalTransmission,
    InfectionToCarcass,
    CarcassDynamics,
    SporeEnvironment,
)

def run_simulation(scavenging_rate=0.0):
    # 1. Define params
    params = PropertySet({
        "nticks": 200,
        "beta": 0.005, # Infection rate per spore
        "mean_infectious_period": 3.0,
        "mean_decomposition_period": 7.0,
        "initial_spores_at_hotspot": 1000.0,
        "spores_per_carcass": 100.0,
        "seed": 42,
    })

    # 2. Define scenario (Grid)
    # 10x10 grid, 100 animals per node
    M, N = 10, 10
    scenario = grid(
        M=M,
        N=N,
        population_fn=lambda r, c: 100
    )

    # Initialize counts
    scenario["S"] = scenario["population"]
    scenario["I"] = 0
    scenario["C"] = 0  # Added Carcass count to ensure it's tracked
    scenario["R"] = 0

    # 3. Build Model
    model = Model(scenario, params)

    # 4. Configure Distributions
    infdist = poisson(params.mean_infectious_period)
    decompdist = poisson(params.mean_decomposition_period)

    # 5. Attach Components
    # Order matters:
    # 1. Susceptible (init, tracking)
    # 2. EnvironmentalTransmission (uses spores[tick], makes new I, sets itimer)
    # 3. InfectionToCarcass (transitions I->C, sets ctimer)
    # 4. CarcassDynamics (transitions C->Removed, produces daily_spores)
    # 5. SporeEnvironment (updates spores[tick+1] using daily_spores)

    model.components = [
        Susceptible(model),
        EnvironmentalTransmission(model, beta=params.beta, infdurdist=infdist),
        InfectionToCarcass(model, decomp_dist=decompdist),
        CarcassDynamics(model, scavenging_rate=scavenging_rate, spores_per_carcass=params.spores_per_carcass),
        SporeEnvironment(model, decay_rate=0.02) # Slow decay
    ]

    # 6. Initialize Spores (Local Infection Zone)
    # Let's put spores in the middle node
    center_node = (M // 2) * N + (N // 2)
    model.nodes.spores[0, center_node] = params.initial_spores_at_hotspot

    # 7. Run
    model.run()

    return model

def plot_population_dynamics(model, title="Simulation Results"):
    """Plots only Infected and Carcass numbers."""
    # Extract total counts
    time = np.arange(model.params.nticks + 1)
    I = model.nodes.I.sum(axis=1)
    C = model.nodes.C.sum(axis=1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(time, I, label="Infected", color='red')
    ax1.plot(time, C, label="Carcass", color='black')
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Population")
    ax1.legend(loc="upper left")

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"laser_environmental_outbreak/{title.replace(' ', '_')}.png")
    plt.close()

    print(f"Plot saved to laser_environmental_outbreak/{title.replace(' ', '_')}.png")

def plot_spore_comparison(models_dict, title="Comparative Spore Dynamics"):
    """Overlays spore count for each scenario."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, model in models_dict.items():
        time = np.arange(model.params.nticks + 1)
        Spores = model.nodes.spores.sum(axis=1)
        ax.plot(time, Spores, label=name)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Total Environmental Spores")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"laser_environmental_outbreak/{title.replace(' ', '_')}.png")
    plt.close()

    print(f"Plot saved to laser_environmental_outbreak/{title.replace(' ', '_')}.png")

def plot_spatial_comparison(models_dict, title="Spatial Spore Distribution (Final Day)"):
    """Spatial heatmaps side-by-side."""
    n_scenarios = len(models_dict)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5), sharey=True)

    # Assume grid is 10x10 as in run_simulation
    M, N = 10, 10

    if n_scenarios == 1:
        axes = [axes]

    # Find global max for consistent colorbar
    max_spores = 0
    for model in models_dict.values():
        max_spores = max(max_spores, model.nodes.spores[-1].max())

    for ax, (name, model) in zip(axes, models_dict.items()):
        # Get final spore state
        final_spores = model.nodes.spores[-1]
        grid_spores = final_spores.reshape((M, N))

        im = ax.imshow(grid_spores, cmap='YlGn', vmin=0, vmax=max_spores, interpolation='nearest')
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("Spore Concentration")

    plt.suptitle(title)
    # plt.tight_layout() # Suptitle sometimes conflicts with tight_layout
    plt.savefig(f"laser_environmental_outbreak/{title.replace(' ', '_')}.png")
    plt.close()

    print(f"Plot saved to laser_environmental_outbreak/{title.replace(' ', '_')}.png")


def main():
    models = {}

    print("Running Baseline (Scavenging = 0.0)...")
    models["Baseline (0.0)"] = run_simulation(scavenging_rate=0.0)
    plot_population_dynamics(models["Baseline (0.0)"], "Baseline Anthrax Outbreak (No Scavenging)")

    print("Running with Scavenging (Rate = 0.2)...")
    models["Scavenging (0.2)"] = run_simulation(scavenging_rate=0.2)
    plot_population_dynamics(models["Scavenging (0.2)"], "Anthrax Outbreak with High Scavenging")

    print("Running with Scavenging (Rate = 0.5)...")
    models["Scavenging (0.5)"] = run_simulation(scavenging_rate=0.5)
    plot_population_dynamics(models["Scavenging (0.5)"], "Anthrax Outbreak with Very High Scavenging")

    # Comparative plots
    print("Generating comparative plots...")
    plot_spore_comparison(models)
    plot_spatial_comparison(models)

if __name__ == "__main__":
    main()
