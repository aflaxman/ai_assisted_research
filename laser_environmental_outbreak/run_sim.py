
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from laser.core import PropertySet
from laser.core.distributions import poisson, constant_int
from laser.generic.model import Model
from laser.generic.components import Susceptible
from laser.core.utils import grid
from laser_environmental_outbreak.components import (
    EnvironmentalTransmission,
    InfectionToCarcass,
    CarcassDynamics,
    SporeEnvironment,
    STATE_CARCASS,
    STATE_REMOVED
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
    scenario["R"] = 0 # Not used but good to clear

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

def plot_results(model, title="Simulation Results"):
    # Extract total counts
    time = np.arange(model.params.nticks + 1)
    S = model.nodes.S.sum(axis=1)
    I = model.nodes.I.sum(axis=1)
    C = model.nodes.C.sum(axis=1)
    Spores = model.nodes.spores.sum(axis=1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(time, S, label="Susceptible", color='blue')
    ax1.plot(time, I, label="Infected", color='red')
    ax1.plot(time, C, label="Carcass", color='black')
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Population")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(time, Spores, label="Total Env. Spores", color='green', linestyle='--')
    ax2.set_ylabel("Spores")
    ax2.legend(loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"laser_environmental_outbreak/{title.replace(' ', '_')}.png")
    plt.close()

    print(f"Plot saved to laser_environmental_outbreak/{title.replace(' ', '_')}.png")

def main():
    print("Running Baseline (Scavenging = 0.0)...")
    model_base = run_simulation(scavenging_rate=0.0)
    plot_results(model_base, "Baseline Anthrax Outbreak (No Scavenging)")

    print("Running with Scavenging (Rate = 0.2)...")
    model_scav = run_simulation(scavenging_rate=0.2)
    plot_results(model_scav, "Anthrax Outbreak with High Scavenging")

    print("Running with Scavenging (Rate = 0.5)...")
    model_scav_high = run_simulation(scavenging_rate=0.5)
    plot_results(model_scav_high, "Anthrax Outbreak with Very High Scavenging")

if __name__ == "__main__":
    main()
