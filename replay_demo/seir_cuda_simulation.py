#!/usr/bin/env python3
"""
CUDA-Accelerated SEIR Disease Simulation on Temporal Contact Networks

This implements GPU-accelerated Monte Carlo simulations using Numba CUDA,
demonstrating the performance benefits of the Replay framework approach.

The key insight: Monte Carlo simulations are embarrassingly parallel -
each simulation run is independent, making this ideal for GPU acceleration.

Requirements:
    - CUDA-capable GPU
    - numba with CUDA support
    - numpy

Usage:
    python seir_cuda_simulation.py [--n-sims N] [--device cpu|gpu]
"""

import numpy as np
import time
import argparse
from typing import Optional
import json

# Try to import CUDA - fall back gracefully if not available
try:
    from numba import cuda, jit
    import numba
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"CUDA available: {cuda.get_current_device().name}")
except ImportError:
    CUDA_AVAILABLE = False
    print("Numba CUDA not available - GPU acceleration disabled")

# JIT-compiled CPU version for fair comparison
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# =============================================================================
# Data Structures (numpy arrays for GPU compatibility)
# =============================================================================

def generate_contact_network(n_nodes: int, n_days: int, contacts_per_day: int,
                             seed: int = 42) -> tuple:
    """
    Generate a temporal contact network as numpy arrays.

    Returns:
        node1_arr: Array of first node in each contact
        node2_arr: Array of second node in each contact
        time_arr: Array of contact times (hours from start)
        duration_arr: Array of contact durations (hours)
    """
    np.random.seed(seed)

    n_contacts = n_days * contacts_per_day

    # Generate random contacts
    node1_arr = np.random.randint(0, n_nodes, size=n_contacts, dtype=np.int32)
    node2_arr = np.random.randint(0, n_nodes, size=n_contacts, dtype=np.int32)

    # Remove self-contacts
    mask = node1_arr != node2_arr
    node1_arr = node1_arr[mask]
    node2_arr = node2_arr[mask]

    n_contacts = len(node1_arr)

    # Generate times (hours from start, spanning n_days * 12 active hours per day)
    day_indices = np.random.randint(0, n_days, size=n_contacts)
    hour_offsets = np.random.randint(0, 12, size=n_contacts)  # 8am-8pm
    time_arr = (day_indices * 24 + 8 + hour_offsets).astype(np.float32)

    # Sort by time for temporal processing
    sort_idx = np.argsort(time_arr)
    node1_arr = node1_arr[sort_idx]
    node2_arr = node2_arr[sort_idx]
    time_arr = time_arr[sort_idx]

    # Random durations (0.1 to 2 hours)
    duration_arr = np.random.uniform(0.1, 2.0, size=n_contacts).astype(np.float32)

    return node1_arr, node2_arr, time_arr, duration_arr


# =============================================================================
# CPU Implementation (with Numba JIT for fair comparison)
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def run_single_simulation_cpu(
        node1_arr, node2_arr, time_arr, duration_arr,
        n_nodes, transmission_prob, incubation_hours, infectious_hours,
        initial_infection_prob, rng_seed
    ):
        """Run a single SEIR simulation on CPU with Numba JIT."""
        np.random.seed(rng_seed)

        # State arrays: 0=S, 1=E, 2=I, 3=R
        states = np.zeros(n_nodes, dtype=np.int32)
        state_change_times = np.zeros(n_nodes, dtype=np.float32)

        # Initialize infections
        for i in range(n_nodes):
            if np.random.random() < initial_infection_prob:
                states[i] = 2  # Infectious
                state_change_times[i] = time_arr[0] if len(time_arr) > 0 else 0

        # Track SEIR counts over time
        n_timesteps = 60  # Fixed number of output timesteps
        time_min = time_arr[0] if len(time_arr) > 0 else 0
        time_max = time_arr[-1] if len(time_arr) > 0 else 720
        dt = (time_max - time_min) / n_timesteps

        seir_history = np.zeros((n_timesteps, 4), dtype=np.int32)

        contact_idx = 0
        n_contacts = len(time_arr)

        for step in range(n_timesteps):
            current_time = time_min + step * dt

            # Update disease progression
            for i in range(n_nodes):
                if states[i] == 1:  # Exposed
                    if current_time - state_change_times[i] >= incubation_hours:
                        states[i] = 2  # -> Infectious
                        state_change_times[i] = current_time
                elif states[i] == 2:  # Infectious
                    if current_time - state_change_times[i] >= infectious_hours:
                        states[i] = 3  # -> Recovered
                        state_change_times[i] = current_time

            # Process contacts up to current time
            while contact_idx < n_contacts and time_arr[contact_idx] <= current_time:
                n1 = node1_arr[contact_idx]
                n2 = node2_arr[contact_idx]
                dur = duration_arr[contact_idx]

                # Check transmission in both directions
                for source, target in [(n1, n2), (n2, n1)]:
                    if states[source] == 2 and states[target] == 0:
                        prob = 1.0 - (1.0 - transmission_prob) ** dur
                        if np.random.random() < prob:
                            states[target] = 1  # -> Exposed
                            state_change_times[target] = time_arr[contact_idx]

                contact_idx += 1

            # Count states
            for i in range(n_nodes):
                seir_history[step, states[i]] += 1

        return seir_history
else:
    def run_single_simulation_cpu(
        node1_arr, node2_arr, time_arr, duration_arr,
        n_nodes, transmission_prob, incubation_hours, infectious_hours,
        initial_infection_prob, rng_seed
    ):
        """Pure Python fallback (slow)."""
        np.random.seed(rng_seed)

        states = np.zeros(n_nodes, dtype=np.int32)
        state_change_times = np.zeros(n_nodes, dtype=np.float32)

        for i in range(n_nodes):
            if np.random.random() < initial_infection_prob:
                states[i] = 2
                state_change_times[i] = time_arr[0] if len(time_arr) > 0 else 0

        n_timesteps = 60
        time_min = time_arr[0] if len(time_arr) > 0 else 0
        time_max = time_arr[-1] if len(time_arr) > 0 else 720
        dt = (time_max - time_min) / n_timesteps

        seir_history = np.zeros((n_timesteps, 4), dtype=np.int32)
        contact_idx = 0
        n_contacts = len(time_arr)

        for step in range(n_timesteps):
            current_time = time_min + step * dt

            for i in range(n_nodes):
                if states[i] == 1:
                    if current_time - state_change_times[i] >= incubation_hours:
                        states[i] = 2
                        state_change_times[i] = current_time
                elif states[i] == 2:
                    if current_time - state_change_times[i] >= infectious_hours:
                        states[i] = 3
                        state_change_times[i] = current_time

            while contact_idx < n_contacts and time_arr[contact_idx] <= current_time:
                n1 = node1_arr[contact_idx]
                n2 = node2_arr[contact_idx]
                dur = duration_arr[contact_idx]

                for source, target in [(n1, n2), (n2, n1)]:
                    if states[source] == 2 and states[target] == 0:
                        prob = 1.0 - (1.0 - transmission_prob) ** dur
                        if np.random.random() < prob:
                            states[target] = 1
                            state_change_times[target] = time_arr[contact_idx]

                contact_idx += 1

            for i in range(n_nodes):
                seir_history[step, states[i]] += 1

        return seir_history


def run_monte_carlo_cpu(node1_arr, node2_arr, time_arr, duration_arr,
                        n_nodes, params, n_sims, base_seed=42):
    """Run multiple simulations on CPU."""
    results = np.zeros((n_sims, 60, 4), dtype=np.int32)

    for sim in range(n_sims):
        results[sim] = run_single_simulation_cpu(
            node1_arr, node2_arr, time_arr, duration_arr,
            n_nodes,
            params['transmission_prob'],
            params['incubation_hours'],
            params['infectious_hours'],
            params['initial_infection_prob'],
            base_seed + sim
        )
        if (sim + 1) % 100 == 0:
            print(f"    CPU: Completed {sim + 1}/{n_sims} simulations")

    return results


# =============================================================================
# CUDA Implementation
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit
    def seir_kernel(
        node1_arr, node2_arr, time_arr, duration_arr,
        n_nodes, n_contacts, n_timesteps, time_min, dt,
        transmission_prob, incubation_hours, infectious_hours,
        initial_infection_prob,
        results, seeds
    ):
        """
        CUDA kernel for parallel SEIR simulations.
        Each thread runs one complete Monte Carlo simulation.
        """
        sim_idx = cuda.grid(1)

        if sim_idx >= results.shape[0]:
            return

        # Initialize thread-local RNG (simple LCG)
        rng_state = seeds[sim_idx]

        def rand():
            nonlocal rng_state
            rng_state = (rng_state * 1103515245 + 12345) & 0x7fffffff
            return rng_state / 2147483647.0

        # Allocate state arrays in local memory
        # Note: For large n_nodes, consider shared memory or multiple kernel launches
        states = cuda.local.array(1000, dtype=numba.int32)  # Max 1000 nodes
        state_change_times = cuda.local.array(1000, dtype=numba.float32)

        # Initialize states
        for i in range(n_nodes):
            states[i] = 0  # Susceptible
            state_change_times[i] = 0.0
            if rand() < initial_infection_prob:
                states[i] = 2  # Infectious
                state_change_times[i] = time_min

        contact_idx = 0

        # Time stepping
        for step in range(n_timesteps):
            current_time = time_min + step * dt

            # Update disease progression
            for i in range(n_nodes):
                if states[i] == 1:  # Exposed
                    if current_time - state_change_times[i] >= incubation_hours:
                        states[i] = 2  # -> Infectious
                        state_change_times[i] = current_time
                elif states[i] == 2:  # Infectious
                    if current_time - state_change_times[i] >= infectious_hours:
                        states[i] = 3  # -> Recovered
                        state_change_times[i] = current_time

            # Process contacts
            while contact_idx < n_contacts and time_arr[contact_idx] <= current_time:
                n1 = node1_arr[contact_idx]
                n2 = node2_arr[contact_idx]
                dur = duration_arr[contact_idx]

                # Transmission n1 -> n2
                if states[n1] == 2 and states[n2] == 0:
                    prob = 1.0 - (1.0 - transmission_prob) ** dur
                    if rand() < prob:
                        states[n2] = 1
                        state_change_times[n2] = time_arr[contact_idx]

                # Transmission n2 -> n1
                if states[n2] == 2 and states[n1] == 0:
                    prob = 1.0 - (1.0 - transmission_prob) ** dur
                    if rand() < prob:
                        states[n1] = 1
                        state_change_times[n1] = time_arr[contact_idx]

                contact_idx += 1

            # Count states
            s_count, e_count, i_count, r_count = 0, 0, 0, 0
            for i in range(n_nodes):
                if states[i] == 0:
                    s_count += 1
                elif states[i] == 1:
                    e_count += 1
                elif states[i] == 2:
                    i_count += 1
                else:
                    r_count += 1

            results[sim_idx, step, 0] = s_count
            results[sim_idx, step, 1] = e_count
            results[sim_idx, step, 2] = i_count
            results[sim_idx, step, 3] = r_count


    def run_monte_carlo_gpu(node1_arr, node2_arr, time_arr, duration_arr,
                            n_nodes, params, n_sims, base_seed=42):
        """Run multiple simulations on GPU using CUDA."""
        n_timesteps = 60
        time_min = time_arr[0] if len(time_arr) > 0 else 0
        time_max = time_arr[-1] if len(time_arr) > 0 else 720
        dt = (time_max - time_min) / n_timesteps

        # Transfer data to GPU
        d_node1 = cuda.to_device(node1_arr)
        d_node2 = cuda.to_device(node2_arr)
        d_time = cuda.to_device(time_arr)
        d_duration = cuda.to_device(duration_arr)

        # Allocate output array on GPU
        results = np.zeros((n_sims, n_timesteps, 4), dtype=np.int32)
        d_results = cuda.to_device(results)

        # Generate random seeds for each simulation
        np.random.seed(base_seed)
        seeds = np.random.randint(1, 2**31 - 1, size=n_sims, dtype=np.int32)
        d_seeds = cuda.to_device(seeds)

        # Configure kernel launch
        threads_per_block = 256
        blocks = (n_sims + threads_per_block - 1) // threads_per_block

        print(f"    GPU: Launching {blocks} blocks x {threads_per_block} threads = {n_sims} simulations")

        # Launch kernel
        seir_kernel[blocks, threads_per_block](
            d_node1, d_node2, d_time, d_duration,
            n_nodes, len(time_arr), n_timesteps, time_min, dt,
            params['transmission_prob'],
            params['incubation_hours'],
            params['infectious_hours'],
            params['initial_infection_prob'],
            d_results, d_seeds
        )

        # Synchronize and copy results back
        cuda.synchronize()
        results = d_results.copy_to_host()

        return results


# =============================================================================
# Benchmarking and Analysis
# =============================================================================

def aggregate_results(results: np.ndarray) -> dict:
    """Aggregate Monte Carlo results into mean and std."""
    n_sims, n_steps, _ = results.shape

    return {
        'n_simulations': n_sims,
        'n_timesteps': n_steps,
        'S_mean': results[:, :, 0].mean(axis=0).tolist(),
        'S_std': results[:, :, 0].std(axis=0).tolist(),
        'E_mean': results[:, :, 1].mean(axis=0).tolist(),
        'E_std': results[:, :, 1].std(axis=0).tolist(),
        'I_mean': results[:, :, 2].mean(axis=0).tolist(),
        'I_std': results[:, :, 2].std(axis=0).tolist(),
        'R_mean': results[:, :, 3].mean(axis=0).tolist(),
        'R_std': results[:, :, 3].std(axis=0).tolist(),
    }


def run_benchmark(n_nodes=50, n_days=30, contacts_per_day=100,
                  n_sims=1000, run_gpu=True, run_cpu=True):
    """Run benchmark comparing CPU and GPU performance."""

    print("=" * 70)
    print("SEIR CUDA Benchmark: CPU vs GPU Performance")
    print("=" * 70)

    # Generate contact network
    print(f"\n[1] Generating contact network...")
    print(f"    Nodes: {n_nodes}, Days: {n_days}, Contacts/day: {contacts_per_day}")

    node1, node2, times, durations = generate_contact_network(
        n_nodes, n_days, contacts_per_day
    )
    print(f"    Total contacts: {len(node1)}")

    # Disease parameters
    params = {
        'transmission_prob': 0.15,
        'incubation_hours': 48.0,    # 2 days
        'infectious_hours': 120.0,   # 5 days
        'initial_infection_prob': 0.06
    }

    print(f"\n[2] Disease parameters:")
    print(f"    Transmission prob: {params['transmission_prob']}")
    print(f"    Incubation: {params['incubation_hours']/24:.1f} days")
    print(f"    Infectious: {params['infectious_hours']/24:.1f} days")

    results = {}

    # CPU benchmark
    if run_cpu:
        print(f"\n[3] Running CPU benchmark ({n_sims} simulations)...")

        # Warmup
        _ = run_monte_carlo_cpu(node1, node2, times, durations, n_nodes, params, 2)

        start = time.perf_counter()
        cpu_results = run_monte_carlo_cpu(node1, node2, times, durations, n_nodes, params, n_sims)
        cpu_time = time.perf_counter() - start

        results['cpu'] = {
            'time': cpu_time,
            'sims_per_second': n_sims / cpu_time,
            'results': aggregate_results(cpu_results)
        }
        print(f"    CPU time: {cpu_time:.2f}s ({n_sims/cpu_time:.1f} sims/sec)")

    # GPU benchmark
    if run_gpu and CUDA_AVAILABLE:
        print(f"\n[4] Running GPU benchmark ({n_sims} simulations)...")

        # Warmup (compile kernel)
        _ = run_monte_carlo_gpu(node1, node2, times, durations, n_nodes, params, 10)

        start = time.perf_counter()
        gpu_results = run_monte_carlo_gpu(node1, node2, times, durations, n_nodes, params, n_sims)
        gpu_time = time.perf_counter() - start

        results['gpu'] = {
            'time': gpu_time,
            'sims_per_second': n_sims / gpu_time,
            'results': aggregate_results(gpu_results)
        }
        print(f"    GPU time: {gpu_time:.2f}s ({n_sims/gpu_time:.1f} sims/sec)")
    elif run_gpu:
        print(f"\n[4] GPU benchmark skipped (CUDA not available)")

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    if 'cpu' in results:
        print(f"\nCPU Performance:")
        print(f"  Time: {results['cpu']['time']:.2f} seconds")
        print(f"  Throughput: {results['cpu']['sims_per_second']:.1f} simulations/second")

    if 'gpu' in results:
        print(f"\nGPU Performance:")
        print(f"  Time: {results['gpu']['time']:.2f} seconds")
        print(f"  Throughput: {results['gpu']['sims_per_second']:.1f} simulations/second")

        if 'cpu' in results:
            speedup = results['cpu']['time'] / results['gpu']['time']
            print(f"\nSpeedup: {speedup:.1f}x faster on GPU")

    return results


def main():
    parser = argparse.ArgumentParser(description='CUDA SEIR Simulation Benchmark')
    parser.add_argument('--n-sims', type=int, default=1000,
                        help='Number of Monte Carlo simulations (default: 1000)')
    parser.add_argument('--n-nodes', type=int, default=50,
                        help='Number of nodes in contact network (default: 50)')
    parser.add_argument('--n-days', type=int, default=30,
                        help='Number of days to simulate (default: 30)')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'both'], default='both',
                        help='Device to run on (default: both)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results (default: benchmark_results.json)')

    args = parser.parse_args()

    run_cpu = args.device in ('cpu', 'both')
    run_gpu = args.device in ('gpu', 'both')

    results = run_benchmark(
        n_nodes=args.n_nodes,
        n_days=args.n_days,
        n_sims=args.n_sims,
        run_cpu=run_cpu,
        run_gpu=run_gpu
    )

    # Save results
    with open(args.output, 'w') as f:
        # Convert numpy types for JSON serialization
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
