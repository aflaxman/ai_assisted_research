#!/usr/bin/env python3
"""
SEIR Disease Simulation on Temporal Contact Networks

This is a Python demonstration of the concepts behind the Replay framework:
https://github.com/HarrisonGreenlee/Replay

Replay is a GPU-accelerated tool for simulating infectious disease spread
across temporal contact networks. This demo implements the core concepts
in pure Python for educational purposes.

SEIR Model States:
- S (Susceptible): Can become infected
- E (Exposed): Infected but not yet infectious (incubation period)
- I (Infectious): Can transmit disease to others
- R (Recovered/Removed): Immune or removed from population
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional
import json


@dataclass
class SEIRParams:
    """Parameters for SEIR disease model."""
    transmission_prob: float = 0.05      # Probability of transmission per contact hour
    incubation_period: float = 5.0       # Days from exposure to infectious
    infectious_period: float = 7.0       # Days of being infectious
    immunity_period: Optional[float] = None  # Days of immunity (None = permanent)
    initial_infection_prob: float = 0.01  # Probability of initial infection


@dataclass
class Contact:
    """A contact between two nodes at a specific time."""
    node1: int
    node2: int
    timestamp: datetime
    duration_hours: float = 1.0


class Node:
    """Individual in the simulation."""

    def __init__(self, node_id: int):
        self.id = node_id
        self.state = 'S'  # S, E, I, R
        self.state_change_time: Optional[datetime] = None
        self.infection_source: Optional[int] = None

    def __repr__(self):
        return f"Node({self.id}, state={self.state})"


class TemporalContactNetwork:
    """
    Temporal contact network where contacts occur at specific times.

    This is the core data structure that Replay operates on - instead of
    a static network, contacts are time-stamped events.
    """

    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.contacts: list[Contact] = []
        self.contacts_by_time: dict[datetime, list[Contact]] = defaultdict(list)

    def add_node(self, node_id: int):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id)

    def add_contact(self, contact: Contact):
        self.add_node(contact.node1)
        self.add_node(contact.node2)
        self.contacts.append(contact)
        self.contacts_by_time[contact.timestamp].append(contact)

    def get_time_range(self) -> tuple[datetime, datetime]:
        if not self.contacts:
            now = datetime.now()
            return now, now
        times = [c.timestamp for c in self.contacts]
        return min(times), max(times)

    def reset_states(self):
        """Reset all nodes to susceptible."""
        for node in self.nodes.values():
            node.state = 'S'
            node.state_change_time = None
            node.infection_source = None


class SEIRSimulation:
    """
    Monte Carlo SEIR simulation on a temporal contact network.

    This implements the core simulation logic of Replay:
    1. Process contacts in temporal order
    2. For each contact, check if transmission occurs
    3. Update node states based on disease progression
    """

    def __init__(self, network: TemporalContactNetwork, params: SEIRParams):
        self.network = network
        self.params = params
        self.history: list[dict] = []

    def _initialize_infections(self, current_time: datetime):
        """Seed initial infections."""
        for node in self.network.nodes.values():
            if random.random() < self.params.initial_infection_prob:
                node.state = 'I'
                node.state_change_time = current_time

    def _update_disease_progression(self, current_time: datetime):
        """Update E->I and I->R transitions based on time."""
        for node in self.network.nodes.values():
            if node.state_change_time is None:
                continue

            days_in_state = (current_time - node.state_change_time).total_seconds() / 86400

            if node.state == 'E' and days_in_state >= self.params.incubation_period:
                node.state = 'I'
                node.state_change_time = current_time

            elif node.state == 'I' and days_in_state >= self.params.infectious_period:
                node.state = 'R'
                node.state_change_time = current_time

            elif node.state == 'R' and self.params.immunity_period is not None:
                if days_in_state >= self.params.immunity_period:
                    node.state = 'S'
                    node.state_change_time = current_time

    def _process_contact(self, contact: Contact):
        """Process a single contact for potential transmission."""
        node1 = self.network.nodes[contact.node1]
        node2 = self.network.nodes[contact.node2]

        # Check transmission in both directions
        for source, target in [(node1, node2), (node2, node1)]:
            if source.state == 'I' and target.state == 'S':
                # Transmission probability scales with contact duration
                prob = 1 - (1 - self.params.transmission_prob) ** contact.duration_hours
                if random.random() < prob:
                    target.state = 'E'
                    target.state_change_time = contact.timestamp
                    target.infection_source = source.id

    def _count_states(self) -> dict[str, int]:
        """Count nodes in each state."""
        counts = {'S': 0, 'E': 0, 'I': 0, 'R': 0}
        for node in self.network.nodes.values():
            counts[node.state] += 1
        return counts

    def run(self, time_step_hours: float = 1.0) -> list[dict]:
        """
        Run a single Monte Carlo simulation.

        Returns a list of state counts at each time step.
        """
        self.network.reset_states()
        self.history = []

        start_time, end_time = self.network.get_time_range()
        self._initialize_infections(start_time)

        current_time = start_time
        step_delta = timedelta(hours=time_step_hours)

        while current_time <= end_time:
            # Update disease progression
            self._update_disease_progression(current_time)

            # Process contacts at this time step
            for contact in self.network.contacts_by_time.get(current_time, []):
                self._process_contact(contact)

            # Record state
            counts = self._count_states()
            counts['time'] = current_time.isoformat()
            self.history.append(counts)

            current_time += step_delta

        return self.history


def run_monte_carlo(network: TemporalContactNetwork,
                    params: SEIRParams,
                    n_simulations: int = 100,
                    time_step_hours: float = 1.0) -> list[list[dict]]:
    """
    Run multiple Monte Carlo simulations.

    This is what Replay accelerates with GPU - running thousands of
    these simulations in parallel.
    """
    results = []
    sim = SEIRSimulation(network, params)

    for i in range(n_simulations):
        history = sim.run(time_step_hours)
        results.append(history)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_simulations} simulations")

    return results


def aggregate_results(results: list[list[dict]]) -> dict:
    """Aggregate Monte Carlo results into mean and std."""
    if not results or not results[0]:
        return {}

    n_steps = len(results[0])
    n_sims = len(results)

    aggregated = {
        'times': [],
        'S_mean': [], 'S_std': [],
        'E_mean': [], 'E_std': [],
        'I_mean': [], 'I_std': [],
        'R_mean': [], 'R_std': [],
    }

    for step in range(n_steps):
        aggregated['times'].append(results[0][step]['time'])

        for state in ['S', 'E', 'I', 'R']:
            values = [results[sim][step][state] for sim in range(n_sims)]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            aggregated[f'{state}_mean'].append(mean)
            aggregated[f'{state}_std'].append(std)

    return aggregated


def generate_sample_network(n_nodes: int = 50,
                            n_days: int = 30,
                            contacts_per_day: int = 100) -> TemporalContactNetwork:
    """
    Generate a sample temporal contact network.

    In real use, Replay ingests actual contact data (e.g., from
    Bluetooth proximity sensors, contact tracing apps, etc.)
    """
    network = TemporalContactNetwork()

    # Add nodes
    for i in range(n_nodes):
        network.add_node(i)

    # Generate random contacts over time
    start_time = datetime(2025, 1, 1, 8, 0, 0)

    for day in range(n_days):
        for _ in range(contacts_per_day):
            node1 = random.randint(0, n_nodes - 1)
            node2 = random.randint(0, n_nodes - 1)
            if node1 == node2:
                continue

            # Random hour during the day (8am - 8pm)
            hour = random.randint(0, 12)
            timestamp = start_time + timedelta(days=day, hours=hour)
            duration = random.uniform(0.1, 2.0)  # 6 min to 2 hours

            contact = Contact(node1, node2, timestamp, duration)
            network.add_contact(contact)

    return network


def print_summary(results: list[list[dict]], aggregated: dict):
    """Print a summary of simulation results."""
    n_sims = len(results)

    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)

    # Final state
    final_idx = -1
    print(f"\nFinal state (averaged over {n_sims} simulations):")
    print(f"  Susceptible: {aggregated['S_mean'][final_idx]:.1f} +/- {aggregated['S_std'][final_idx]:.1f}")
    print(f"  Exposed:     {aggregated['E_mean'][final_idx]:.1f} +/- {aggregated['E_std'][final_idx]:.1f}")
    print(f"  Infectious:  {aggregated['I_mean'][final_idx]:.1f} +/- {aggregated['I_std'][final_idx]:.1f}")
    print(f"  Recovered:   {aggregated['R_mean'][final_idx]:.1f} +/- {aggregated['R_std'][final_idx]:.1f}")

    # Peak infections
    peak_I = max(aggregated['I_mean'])
    peak_idx = aggregated['I_mean'].index(peak_I)
    print(f"\nPeak infectious: {peak_I:.1f} at time step {peak_idx}")

    # Attack rate
    total_nodes = aggregated['S_mean'][0] + aggregated['R_mean'][0]
    attack_rate = aggregated['R_mean'][final_idx] / total_nodes * 100
    print(f"Attack rate: {attack_rate:.1f}%")


def save_results(aggregated: dict, filename: str = "simulation_results.json"):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    print("="*60)
    print("SEIR Disease Simulation on Temporal Contact Networks")
    print("Demonstrating concepts from the Replay framework")
    print("https://github.com/HarrisonGreenlee/Replay")
    print("="*60)

    # Set random seed for reproducibility
    random.seed(42)

    # Generate sample network
    print("\n[1] Generating temporal contact network...")
    network = generate_sample_network(
        n_nodes=50,
        n_days=30,
        contacts_per_day=100
    )
    print(f"    Created network with {len(network.nodes)} nodes and {len(network.contacts)} contacts")

    # Configure disease parameters (tuned for visible epidemic dynamics)
    print("\n[2] Configuring SEIR parameters...")
    params = SEIRParams(
        transmission_prob=0.15,       # 15% transmission per contact hour
        incubation_period=2.0,        # 2 days incubation
        infectious_period=5.0,        # 5 days infectious
        immunity_period=None,         # Permanent immunity
        initial_infection_prob=0.06   # 6% initially infected (~3 people)
    )
    print(f"    Transmission probability: {params.transmission_prob}")
    print(f"    Incubation period: {params.incubation_period} days")
    print(f"    Infectious period: {params.infectious_period} days")

    # Run Monte Carlo simulations
    print("\n[3] Running Monte Carlo simulations...")
    n_sims = 50
    print(f"    Running {n_sims} simulations (Replay does 1000s on GPU)...")
    results = run_monte_carlo(network, params, n_simulations=n_sims, time_step_hours=12.0)

    # Aggregate results
    print("\n[4] Aggregating results...")
    aggregated = aggregate_results(results)

    # Print summary
    print_summary(results, aggregated)

    # Save results
    save_results(aggregated, "simulation_results.json")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("\nNote: The actual Replay tool uses CUDA to run thousands of")
    print("simulations in parallel, achieving much higher performance.")
    print("This Python demo is for educational purposes to understand")
    print("the core concepts of SEIR simulation on contact networks.")


if __name__ == "__main__":
    main()
