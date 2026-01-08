"""
Utility functions for the LASER environmental outbreak simulation.

This module extracts common patterns to avoid code duplication (DRY principle).
Each function is documented to help novice readers understand what it does.
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt


# ==============================================================================
# PROPERTY INITIALIZATION HELPERS
# ==============================================================================

def ensure_vector_property(model, collection_name, property_name, dtype):
    """
    Safely add a vector property to a model collection if it doesn't exist.

    A 'vector property' stores values over time (one value per time tick).
    This is useful for tracking how things change during the simulation.

    Args:
        model: The simulation model
        collection_name: Either 'nodes' or 'people'
        property_name: Name of the property to create (e.g., 'spores', 'daily_spores')
        dtype: Data type (e.g., np.float32, np.int32)

    Example:
        ensure_vector_property(model, 'nodes', 'spores', np.float32)
        # Now model.nodes.spores exists and can track spore counts over time
    """
    collection = getattr(model, collection_name)
    if not hasattr(collection, property_name):
        collection.add_vector_property(property_name, model.params.nticks + 1, dtype=dtype)


def ensure_scalar_property(model, collection_name, property_name, dtype):
    """
    Safely add a scalar property to a model collection if it doesn't exist.

    A 'scalar property' stores a single value per entity (person or node).
    This is useful for tracking individual-level attributes like timers.

    Args:
        model: The simulation model
        collection_name: Either 'nodes' or 'people'
        property_name: Name of the property to create (e.g., 'itimer', 'ctimer')
        dtype: Data type (e.g., np.uint16, np.int32)

    Example:
        ensure_scalar_property(model, 'people', 'itimer', np.uint16)
        # Now model.people.itimer exists for tracking infection timers
    """
    collection = getattr(model, collection_name)
    if not hasattr(collection, property_name):
        collection.add_scalar_property(property_name, dtype=dtype)


# ==============================================================================
# THREAD-SAFE AGGREGATION HELPERS
# ==============================================================================

def create_thread_safe_array(model, dtype):
    """
    Create an array for thread-safe parallel operations.

    When using parallel computing (numba), each thread needs its own workspace
    to avoid conflicts. This creates a 2D array where:
    - Each row belongs to one thread
    - Each column represents one node in the simulation

    Args:
        model: The simulation model
        dtype: Data type for the array (e.g., np.int32, np.float32)

    Returns:
        A 2D numpy array of shape (num_threads, num_nodes)

    Example:
        newly_infected = create_thread_safe_array(model, np.int32)
        # Each thread can safely write to its own row
    """
    return np.zeros((nb.get_num_threads(), model.nodes.count), dtype=dtype)


def aggregate_thread_results(thread_array, target_dtype=None):
    """
    Combine results from all threads into a single array.

    After parallel computation, each thread has written to its own row.
    This function adds up all rows to get the total results across threads.

    Args:
        thread_array: 2D array with shape (num_threads, num_nodes)
        target_dtype: Optional data type for the output (defaults to input dtype)

    Returns:
        1D array with shape (num_nodes,) containing the sum across threads

    Example:
        newly_infected = create_thread_safe_array(model, np.int32)
        # ... parallel computation writes to newly_infected ...
        total_infected = aggregate_thread_results(newly_infected)
        # Now total_infected[i] contains total new infections at node i
    """
    result = thread_array.sum(axis=0)
    if target_dtype is not None:
        result = result.astype(target_dtype)
    return result


# ==============================================================================
# STATE PROPAGATION HELPER
# ==============================================================================

def propagate_state_counts(model, tick, *state_names):
    """
    Copy state counts from current tick to next tick.

    At the start of each time step, we need to carry forward the counts
    from the previous time step. This function copies multiple state counts
    at once to avoid repetitive code.

    Args:
        model: The simulation model
        tick: Current time tick
        *state_names: Names of states to propagate (e.g., 'I', 'C', 'S')

    Example:
        propagate_state_counts(model, tick, 'I', 'C')
        # This copies both I and C counts from tick to tick+1:
        # model.nodes.I[tick+1] = model.nodes.I[tick]
        # model.nodes.C[tick+1] = model.nodes.C[tick]
    """
    for state_name in state_names:
        state_array = getattr(model.nodes, state_name)
        state_array[tick + 1] = state_array[tick]


# ==============================================================================
# PLOTTING HELPERS
# ==============================================================================

def save_and_close_plot(title, verbose=True, tight_layout=True):
    """
    Save the current plot to a file and close it.

    This standardizes how we save plots across all plotting functions.
    The filename is created by replacing spaces with underscores.

    Args:
        title: Title for the plot (also used to create filename)
        verbose: If True, print a message about where the file was saved
        tight_layout: If True, apply plt.tight_layout() before saving

    Returns:
        The filename where the plot was saved

    Example:
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.title("My Results")
        save_and_close_plot("My Results")
        # Saves to "My_Results.png" and closes the plot
    """
    filename = f"{title.replace(' ', '_')}.png"

    if tight_layout:
        plt.tight_layout()

    plt.savefig(filename)
    plt.close()

    if verbose:
        print(f"Plot saved to {filename}")

    return filename
