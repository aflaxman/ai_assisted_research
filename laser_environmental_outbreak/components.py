"""
Custom components for environmental anthrax outbreak simulation.

This module defines the disease dynamics for an anthrax outbreak model
where animals get infected from environmental spores, die and become carcasses,
and those carcasses release more spores into the environment.

The simulation uses these states:
- Susceptible (S): Healthy animals that can get infected
- Infectious (I): Infected animals (still alive but sick)
- Carcass (C): Dead animals decomposing
- Removed (R): Fully decomposed or scavenged carcasses
"""

import numpy as np
import numba as nb
from laser.generic.shared import State
from laser.generic.components import TransmissionSI, nb_timer_update_timer_set
from utils import (
    ensure_vector_property,
    ensure_scalar_property,
    create_thread_safe_array,
    aggregate_thread_results,
    propagate_state_counts,
)

# Define custom states beyond the standard LASER states
STATE_CARCASS = 4  # Dead animals that may release spores
STATE_REMOVED = 5  # Fully decomposed or scavenged carcasses

class SporeEnvironment:
    """
    Manages environmental spores in the simulation.

    Spores persist in the environment and decay slowly over time.
    New spores are added when carcasses decompose.

    The component tracks:
    - spores[tick]: Total spores in environment at each time step
    - daily_spores[tick]: New spores added during this time step

    Args:
        model: The simulation model
        decay_rate: Fraction of spores that decay each day (0.0 = no decay, 1.0 = all decay)
    """
    def __init__(self, model, decay_rate=0.0):
        self.model = model
        self.decay_rate = decay_rate

        # Ensure the model can track spore counts over time
        ensure_vector_property(model, 'nodes', 'spores', np.float32)
        ensure_vector_property(model, 'nodes', 'daily_spores', np.float32)

    def step(self, tick):
        # Update spores for next tick
        # spores[t+1] = spores[t] * (1 - decay) + daily_spores[t]
        current = self.model.nodes.spores[tick]
        new_additions = self.model.nodes.daily_spores[tick]

        remaining = current * (1.0 - self.decay_rate)
        self.model.nodes.spores[tick + 1] = remaining + new_additions

class EnvironmentalTransmission:
    """
    Models infection from environmental spores (S -> I transition).

    Animals become infected when exposed to spores in their environment.
    The risk of infection increases with spore concentration.

    The force of infection (risk) is calculated as:
        lambda = beta * spores
    where beta is the infection rate parameter.

    Args:
        model: The simulation model
        beta: Infection rate per spore (controls how infectious spores are)
        infdurdist: Probability distribution for infection duration
        infdurmin: Minimum infection duration (days)
    """
    def __init__(self, model, beta, infdurdist, infdurmin=1):
        self.model = model
        self.beta = beta
        self.infdurdist = infdurdist
        self.infdurmin = infdurmin

        # Set up properties to track infections over time
        ensure_vector_property(model, 'nodes', 'I', np.int32)
        ensure_vector_property(model, 'nodes', 'newly_infected', np.int32)
        ensure_vector_property(model, 'nodes', 'spores', np.float32)

        # Track individual infection timers (when will each infected animal die?)
        ensure_scalar_property(model, 'people', 'itimer', np.uint16)

    def step(self, tick):
        """Execute one time step of environmental transmission."""
        spores = self.model.nodes.spores[tick]

        # Calculate force of infection (risk of getting infected)
        # Higher spore concentration = higher risk
        force_of_infection = self.beta * spores

        # Convert force to probability using exponential formula
        # This prevents probabilities from exceeding 1.0
        infection_probability = -np.expm1(-force_of_infection)

        # Create thread-safe array for parallel computation
        newly_infected_by_node = create_thread_safe_array(self.model, np.int32)

        # Apply infections using LASER's built-in parallel transmission function
        TransmissionSI.nb_transmission_step(
            self.model.people.state,
            self.model.people.nodeid,
            infection_probability,
            newly_infected_by_node,
            self.model.people.itimer,
            self.infdurdist,
            self.infdurmin,
            tick,
        )

        # Combine results from all threads
        newly_infected_by_node = aggregate_thread_results(
            newly_infected_by_node, target_dtype=self.model.nodes.S.dtype
        )

        # Carry forward the infection count from previous tick
        propagate_state_counts(self.model, tick, 'I')

        # Update population counts for this time step
        self.model.nodes.S[tick + 1] -= newly_infected_by_node
        self.model.nodes.I[tick + 1] += newly_infected_by_node
        self.model.nodes.newly_infected[tick] = newly_infected_by_node

class InfectionToCarcass:
    """
    Models death of infected animals (I -> Carcass transition).

    Infected animals die after their infection timer runs out.
    When they die, they become carcasses that will decompose.

    Args:
        model: The simulation model
        decomp_dist: Probability distribution for decomposition duration
        decomp_min: Minimum decomposition duration (days)
    """
    def __init__(self, model, decomp_dist, decomp_min=1):
        self.model = model
        self.decomp_dist = decomp_dist
        self.decomp_min = decomp_min

        # Track infection and decomposition timers
        ensure_scalar_property(model, 'people', 'itimer', np.uint16)
        ensure_scalar_property(model, 'people', 'ctimer', np.uint16)

        # Track carcass counts over time
        ensure_vector_property(model, 'nodes', 'C', np.int32)
        ensure_vector_property(model, 'nodes', 'newly_carcass', np.int32)

    def step(self, tick):
        """Execute one time step of infection-to-carcass transitions."""
        # Create thread-safe array for parallel computation
        newly_carcass_by_node = create_thread_safe_array(self.model, np.int32)

        # Transition infected animals with expired timers to carcass state
        # This LASER helper function:
        # 1. Checks each infected animal's timer (itimer)
        # 2. If timer = 0, transitions them to carcass state
        # 3. Sets a new timer (ctimer) for decomposition duration
        nb_timer_update_timer_set(
            self.model.people.state,
            State.INFECTIOUS.value,  # Current state to check
            self.model.people.itimer,  # Timer to check
            STATE_CARCASS,  # New state to transition to
            self.model.people.ctimer,  # New timer to set
            newly_carcass_by_node,  # Output array
            self.model.people.nodeid,
            self.decomp_dist,
            self.decomp_min,
            tick
        )

        # Combine results from all threads
        newly_carcass_by_node = aggregate_thread_results(
            newly_carcass_by_node, target_dtype=self.model.nodes.S.dtype
        )

        # Carry forward counts and update
        propagate_state_counts(self.model, tick, 'C')
        self.model.nodes.I[tick + 1] -= newly_carcass_by_node
        self.model.nodes.C[tick + 1] += newly_carcass_by_node
        self.model.nodes.newly_carcass[tick] = newly_carcass_by_node

@nb.njit(nogil=True, parallel=True)
def nb_carcass_step(states, ctimers, nodeids, scavenging_rate, spores_per_carcass,
                   daily_spores, newly_removed_by_node, newly_scavenged_by_node):
    """
    Parallel function to process carcass dynamics.

    For each carcass, two things can happen:
    1. Scavenging: Carcass is eaten/removed (no spores released)
    2. Decomposition: Carcass decomposes naturally (releases spores)

    This function is compiled with Numba for high performance.
    The 'parallel=True' flag means it runs on multiple CPU cores.
    """
    for i in nb.prange(len(states)):
        if states[i] == STATE_CARCASS:
            nid = nodeids[i]

            # First check: Can this carcass be scavenged?
            if scavenging_rate > 0:
                if np.random.rand() < scavenging_rate:
                    # Carcass was scavenged (eaten by scavengers)
                    # No spores released!
                    states[i] = STATE_REMOVED
                    newly_removed_by_node[nb.get_thread_id(), nid] += 1
                    newly_scavenged_by_node[nb.get_thread_id(), nid] += 1
                    continue  # Skip decomposition

            # Second check: Has decomposition timer expired?
            ctimers[i] -= 1
            if ctimers[i] == 0:
                # Carcass fully decomposed
                # Release spores into the environment!
                states[i] = STATE_REMOVED
                daily_spores[nb.get_thread_id(), nid] += spores_per_carcass
                newly_removed_by_node[nb.get_thread_id(), nid] += 1

class CarcassDynamics:
    """
    Models carcass removal through scavenging or decomposition.

    Carcasses can exit the system in two ways:
    1. Scavenging: Eaten by scavengers (no spore release)
    2. Decomposition: Natural decay (releases spores)

    This creates a feedback loop:
    - More decomposition → more spores → more infections
    - More scavenging → fewer spores → fewer infections

    Args:
        model: The simulation model
        scavenging_rate: Daily probability a carcass is scavenged (0.0 to 1.0)
        spores_per_carcass: Number of spores released when a carcass decomposes
    """
    def __init__(self, model, scavenging_rate, spores_per_carcass):
        self.model = model
        self.scavenging_rate = scavenging_rate
        self.spores_per_carcass = spores_per_carcass

        # Track removal events
        ensure_vector_property(model, 'nodes', 'newly_removed', np.int32)
        ensure_vector_property(model, 'nodes', 'newly_scavenged', np.int32)

        # Track spore production
        ensure_vector_property(model, 'nodes', 'daily_spores', np.float32)

    def step(self, tick):
        """Execute one time step of carcass dynamics."""
        # Create thread-safe arrays for parallel computation
        newly_removed_by_node = create_thread_safe_array(self.model, np.int32)
        newly_scavenged_by_node = create_thread_safe_array(self.model, np.int32)
        daily_spores_thread = create_thread_safe_array(self.model, np.float32)

        # Process all carcasses in parallel
        nb_carcass_step(
            self.model.people.state,
            self.model.people.ctimer,
            self.model.people.nodeid,
            self.scavenging_rate,
            self.spores_per_carcass,
            daily_spores_thread,
            newly_removed_by_node,
            newly_scavenged_by_node
        )

        # Combine results from all threads
        newly_removed = aggregate_thread_results(newly_removed_by_node, target_dtype=self.model.nodes.S.dtype)
        newly_scavenged = aggregate_thread_results(newly_scavenged_by_node, target_dtype=self.model.nodes.S.dtype)
        daily_spores = aggregate_thread_results(daily_spores_thread)

        # Update carcass counts
        self.model.nodes.C[tick + 1] -= newly_removed
        self.model.nodes.newly_removed[tick] = newly_removed
        self.model.nodes.newly_scavenged[tick] = newly_scavenged

        # Add newly released spores to the daily total
        self.model.nodes.daily_spores[tick] += daily_spores
