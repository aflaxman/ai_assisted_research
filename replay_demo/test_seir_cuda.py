#!/usr/bin/env python3
"""
Test suite for SEIR CUDA Simulation

Tests correctness of CPU and GPU implementations and measures speedup.

Usage:
    python test_seir_cuda.py           # Run all tests
    python test_seir_cuda.py -v        # Verbose output
    pytest test_seir_cuda.py -v        # Using pytest
"""

import numpy as np
import time
import sys

# Import the simulation module
from seir_cuda_simulation import (
    generate_contact_network,
    run_single_simulation_cpu,
    run_monte_carlo_cpu,
    aggregate_results,
    CUDA_AVAILABLE,
    NUMBA_AVAILABLE
)

if CUDA_AVAILABLE:
    from seir_cuda_simulation import run_monte_carlo_gpu


# =============================================================================
# Test Configuration
# =============================================================================

DEFAULT_PARAMS = {
    'transmission_prob': 0.15,
    'incubation_hours': 48.0,
    'infectious_hours': 120.0,
    'initial_infection_prob': 0.06
}


# =============================================================================
# Unit Tests
# =============================================================================

def test_contact_network_generation():
    """Test that contact network generation produces valid data."""
    print("\n[TEST] Contact network generation...")

    n_nodes, n_days, cpd = 50, 30, 100
    node1, node2, times, durations = generate_contact_network(n_nodes, n_days, cpd)

    # Check array types
    assert node1.dtype == np.int32, "node1 should be int32"
    assert node2.dtype == np.int32, "node2 should be int32"
    assert times.dtype == np.float32, "times should be float32"
    assert durations.dtype == np.float32, "durations should be float32"

    # Check value ranges
    assert node1.min() >= 0 and node1.max() < n_nodes, "node1 out of range"
    assert node2.min() >= 0 and node2.max() < n_nodes, "node2 out of range"
    assert (node1 != node2).all(), "Self-contacts should be removed"
    assert durations.min() > 0, "Durations should be positive"

    # Check time ordering
    assert np.all(np.diff(times) >= 0), "Times should be sorted"

    print(f"    Generated {len(node1)} contacts for {n_nodes} nodes over {n_days} days")
    print("    PASSED")
    return True


def test_single_simulation_cpu():
    """Test that a single CPU simulation produces valid results."""
    print("\n[TEST] Single CPU simulation...")

    n_nodes = 50
    node1, node2, times, durations = generate_contact_network(n_nodes, 30, 100)

    result = run_single_simulation_cpu(
        node1, node2, times, durations,
        n_nodes,
        DEFAULT_PARAMS['transmission_prob'],
        DEFAULT_PARAMS['incubation_hours'],
        DEFAULT_PARAMS['infectious_hours'],
        DEFAULT_PARAMS['initial_infection_prob'],
        42  # seed
    )

    # Check output shape
    assert result.shape == (60, 4), f"Expected (60, 4), got {result.shape}"

    # Check conservation: total should always equal n_nodes
    totals = result.sum(axis=1)
    assert np.all(totals == n_nodes), "Total population should be conserved"

    # Check no negative counts
    assert result.min() >= 0, "State counts should be non-negative"

    # Check initial state makes sense (should have some S and possibly I)
    assert result[0, 0] > 0, "Should have susceptible individuals"

    print(f"    Simulation produced {result.shape[0]} timesteps")
    print(f"    Initial state: S={result[0,0]}, E={result[0,1]}, I={result[0,2]}, R={result[0,3]}")
    print(f"    Final state: S={result[-1,0]}, E={result[-1,1]}, I={result[-1,2]}, R={result[-1,3]}")
    print("    PASSED")
    return True


def test_monte_carlo_cpu():
    """Test Monte Carlo simulation on CPU."""
    print("\n[TEST] Monte Carlo CPU simulation...")

    n_nodes = 50
    n_sims = 20
    node1, node2, times, durations = generate_contact_network(n_nodes, 30, 100)

    results = run_monte_carlo_cpu(
        node1, node2, times, durations,
        n_nodes, DEFAULT_PARAMS, n_sims
    )

    assert results.shape == (n_sims, 60, 4), f"Expected ({n_sims}, 60, 4), got {results.shape}"

    # Check all simulations conserve population
    for sim in range(n_sims):
        totals = results[sim].sum(axis=1)
        assert np.all(totals == n_nodes), f"Simulation {sim} doesn't conserve population"

    print(f"    Ran {n_sims} simulations successfully")
    print("    PASSED")
    return True


def test_monte_carlo_gpu():
    """Test Monte Carlo simulation on GPU."""
    if not CUDA_AVAILABLE:
        print("\n[TEST] Monte Carlo GPU simulation... SKIPPED (no CUDA)")
        return True

    print("\n[TEST] Monte Carlo GPU simulation...")

    n_nodes = 50
    n_sims = 100
    node1, node2, times, durations = generate_contact_network(n_nodes, 30, 100)

    results = run_monte_carlo_gpu(
        node1, node2, times, durations,
        n_nodes, DEFAULT_PARAMS, n_sims
    )

    assert results.shape == (n_sims, 60, 4), f"Expected ({n_sims}, 60, 4), got {results.shape}"

    # Check all simulations conserve population
    for sim in range(n_sims):
        totals = results[sim].sum(axis=1)
        assert np.all(totals == n_nodes), f"GPU Simulation {sim} doesn't conserve population"

    print(f"    Ran {n_sims} GPU simulations successfully")
    print("    PASSED")
    return True


def test_cpu_gpu_consistency():
    """Test that CPU and GPU produce statistically similar results."""
    if not CUDA_AVAILABLE:
        print("\n[TEST] CPU/GPU consistency... SKIPPED (no CUDA)")
        return True

    print("\n[TEST] CPU/GPU consistency...")

    n_nodes = 50
    n_sims = 500  # Need enough for statistical comparison
    node1, node2, times, durations = generate_contact_network(n_nodes, 30, 100, seed=123)

    # Run CPU
    cpu_results = run_monte_carlo_cpu(
        node1, node2, times, durations,
        n_nodes, DEFAULT_PARAMS, n_sims, base_seed=456
    )
    cpu_agg = aggregate_results(cpu_results)

    # Run GPU
    gpu_results = run_monte_carlo_gpu(
        node1, node2, times, durations,
        n_nodes, DEFAULT_PARAMS, n_sims, base_seed=789
    )
    gpu_agg = aggregate_results(gpu_results)

    # Compare final states (should be statistically similar)
    # Using loose tolerance since different RNG implementations
    for state in ['S', 'E', 'I', 'R']:
        cpu_mean = cpu_agg[f'{state}_mean'][-1]
        gpu_mean = gpu_agg[f'{state}_mean'][-1]
        cpu_std = cpu_agg[f'{state}_std'][-1]

        # Allow 2 standard deviations difference
        tolerance = max(2 * cpu_std, 2.0)
        diff = abs(cpu_mean - gpu_mean)

        assert diff < tolerance, f"{state}: CPU={cpu_mean:.1f}, GPU={gpu_mean:.1f}, diff={diff:.1f} > tol={tolerance:.1f}"
        print(f"    {state}: CPU={cpu_mean:.1f}, GPU={gpu_mean:.1f} (diff={diff:.2f})")

    print("    PASSED")
    return True


def test_speedup():
    """Benchmark CPU vs GPU speedup."""
    print("\n[TEST] Performance benchmark...")

    n_nodes = 50
    n_sims = 500
    node1, node2, times, durations = generate_contact_network(n_nodes, 30, 100)

    # CPU benchmark
    if NUMBA_AVAILABLE:
        # Warmup for JIT
        _ = run_monte_carlo_cpu(node1, node2, times, durations, n_nodes, DEFAULT_PARAMS, 5)

    start = time.perf_counter()
    _ = run_monte_carlo_cpu(node1, node2, times, durations, n_nodes, DEFAULT_PARAMS, n_sims)
    cpu_time = time.perf_counter() - start
    cpu_rate = n_sims / cpu_time

    print(f"    CPU: {cpu_time:.2f}s for {n_sims} sims ({cpu_rate:.1f} sims/sec)")

    if CUDA_AVAILABLE:
        # GPU warmup (kernel compilation)
        _ = run_monte_carlo_gpu(node1, node2, times, durations, n_nodes, DEFAULT_PARAMS, 10)

        start = time.perf_counter()
        _ = run_monte_carlo_gpu(node1, node2, times, durations, n_nodes, DEFAULT_PARAMS, n_sims)
        gpu_time = time.perf_counter() - start
        gpu_rate = n_sims / gpu_time

        speedup = cpu_time / gpu_time

        print(f"    GPU: {gpu_time:.2f}s for {n_sims} sims ({gpu_rate:.1f} sims/sec)")
        print(f"    Speedup: {speedup:.1f}x")

        # GPU should provide meaningful speedup for this workload
        if speedup < 1.0:
            print("    WARNING: GPU slower than CPU (expected for small workloads)")
    else:
        print("    GPU: SKIPPED (no CUDA)")

    print("    PASSED")
    return True


def test_large_scale_gpu():
    """Test GPU with larger scale simulation."""
    if not CUDA_AVAILABLE:
        print("\n[TEST] Large-scale GPU test... SKIPPED (no CUDA)")
        return True

    print("\n[TEST] Large-scale GPU test...")

    n_nodes = 200
    n_sims = 5000
    node1, node2, times, durations = generate_contact_network(n_nodes, 60, 300)

    print(f"    Nodes: {n_nodes}, Contacts: {len(node1)}, Simulations: {n_sims}")

    # Warmup
    _ = run_monte_carlo_gpu(node1, node2, times, durations, n_nodes, DEFAULT_PARAMS, 10)

    start = time.perf_counter()
    results = run_monte_carlo_gpu(node1, node2, times, durations, n_nodes, DEFAULT_PARAMS, n_sims)
    elapsed = time.perf_counter() - start

    assert results.shape == (n_sims, 60, 4)

    rate = n_sims / elapsed
    print(f"    Time: {elapsed:.2f}s ({rate:.0f} sims/sec)")
    print("    PASSED")
    return True


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("SEIR CUDA Simulation Test Suite")
    print("=" * 70)

    print(f"\nEnvironment:")
    print(f"  Numba JIT available: {NUMBA_AVAILABLE}")
    print(f"  CUDA available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        from numba import cuda
        print(f"  GPU: {cuda.get_current_device().name}")

    tests = [
        test_contact_network_generation,
        test_single_simulation_cpu,
        test_monte_carlo_cpu,
        test_monte_carlo_gpu,
        test_cpu_gpu_consistency,
        test_speedup,
        test_large_scale_gpu,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"    FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    success = run_all_tests()
    sys.exit(0 if success else 1)
