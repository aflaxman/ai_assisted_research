# Environmental Anthrax Simulation (LASER)

This directory contains a demonstration of using the LASER spatial simulation framework to simulate an environmental outbreak (like Anthrax) with a local infection zone.

The model tracks:
- **Susceptible (S)**: Healthy livestock.
- **Infected (I)**: Animals infected by environmental spores.
- **Carcass (C)**: Dead animals that are decomposing.
- **Removed (R)**: Scavenged or fully decomposed remains.
- **Spores (Env)**: Environmental contamination that drives infection.

## Files

- `components.py`: Custom LASER components for environmental transmission, carcass decomposition, and spore dynamics.
- `run_sim.py`: Script to run the simulation, varying the scavenging rate, and plotting results.

## Model Dynamics

1.  **Environmental Transmission**:
    -   Force of infection ($\lambda$) is proportional to local spore count.
    -   $S \rightarrow I$ transition.
2.  **Disease Progression**:
    -   Infected animals die after an infectious period ($I \rightarrow C$).
3.  **Carcass Dynamics**:
    -   Carcasses ($C$) persist for a decomposition period.
    -   **Scavenging**: Each day, a carcass has a probability of being scavenged (removed without producing spores).
    -   **Decomposition**: If not scavenged, the carcass decomposes and releases a burst of spores into the environment ($C \rightarrow R + Spores$).
4.  **Spore Decay**:
    -   Environmental spores decay slowly over time.

## Setting up the Environment (Conda)

To run this simulation, you need a Python environment with `laser-generic` installed. Here is how to set it up using Conda:

1.  **Create a new Conda environment:**
    ```bash
    conda create -n laser_env python=3.10
    ```

2.  **Activate the environment:**
    ```bash
    conda activate laser_env
    ```

3.  **Install dependencies:**
    Since `laser-generic` is available via PyPI, use pip to install it:
    ```bash
    pip install laser-generic
    ```

## Running the Simulation

Once your environment is set up and activated:

```bash
python laser_environmental_outbreak/run_sim.py
```

This will generate three plots:
1. `Baseline_Anthrax_Outbreak_(No_Scavenging).png`: High spore accumulation.
2. `Anthrax_Outbreak_with_High_Scavenging.png`: Reduced spore accumulation due to scavenging.
3. `Anthrax_Outbreak_with_Very_High_Scavenging.png`: Further reduction.

## Scenarios

The script runs three scenarios:
- **Baseline**: Scavenging rate = 0.0 (All carcasses decompose and release spores).
- **High Scavenging**: Scavenging rate = 0.2 per day.
- **Very High Scavenging**: Scavenging rate = 0.5 per day.

The results demonstrate how scavenging can effectively control an environmental outbreak by preventing spore release.
