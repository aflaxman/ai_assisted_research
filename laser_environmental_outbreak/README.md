# Environmental Anthrax Simulation (LASER)

This directory contains a demonstration of using the LASER spatial simulation framework to simulate an environmental outbreak (like Anthrax) with a local infection zone.

The model tracks:
- **Susceptible (S)**: Healthy livestock.
- **Infected (I)**: Animals infected by environmental spores.
- **Carcass (C)**: Dead animals that are decomposing.
- **Removed (R)**: Scavenged or fully decomposed remains.
- **Spores (Env)**: Environmental contamination that drives infection.

## Files

- **`run_sim.py`**: Main simulation script that sets up scenarios, runs simulations with different scavenging rates, and generates comparison plots. **Start here** to understand how the simulation works!
- **`components.py`**: Custom LASER components for environmental transmission, carcass decomposition, and spore dynamics. Each component is well-documented to guide novice readers.
- **`utils.py`**: Reusable helper functions that eliminate code duplication (DRY principle) while keeping the code easy to understand. Includes helpers for:
  - Property initialization
  - Thread-safe parallel computing
  - Plot saving and formatting

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

Once your environment is set up and activated, navigate to the directory and run the script:

```bash
cd laser_environmental_outbreak
python run_sim.py
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

## Code Design: DRY (Don't Repeat Yourself) Principles

This codebase has been refactored to follow best practices while remaining accessible to novice readers:

### Eliminated Redundancy

**Before refactoring**, similar patterns were repeated throughout the code:
```python
# Repeated in every component initialization:
if not hasattr(self.model.nodes, 'spores'):
    self.model.nodes.add_vector_property("spores", model.params.nticks + 1, dtype=np.float32)
if not hasattr(self.model.nodes, 'daily_spores'):
    self.model.nodes.add_vector_property("daily_spores", model.params.nticks + 1, dtype=np.float32)
```

**After refactoring**, we use clear helper functions:
```python
# Simple, reusable helpers:
ensure_vector_property(model, 'nodes', 'spores', np.float32)
ensure_vector_property(model, 'nodes', 'daily_spores', np.float32)
```

### Key Improvements

1. **Property Initialization**: `ensure_vector_property()` and `ensure_scalar_property()` replace repetitive if/hasattr checks
2. **Thread-Safe Computing**: `create_thread_safe_array()` and `aggregate_thread_results()` standardize parallel computing patterns
3. **State Propagation**: `propagate_state_counts()` handles the common pattern of carrying forward counts between time steps
4. **Plotting**: `save_and_close_plot()` standardizes how plots are saved and closed

### Benefits

- **Less code duplication**: Changes only need to be made in one place
- **Easier to understand**: Descriptive function names explain what the code does
- **Well-documented**: Each helper function includes docstrings with examples
- **Novice-friendly**: Comments and documentation guide readers through complex concepts like parallel computing

The refactoring makes the code both **more maintainable for experts** and **more accessible for learners**!
