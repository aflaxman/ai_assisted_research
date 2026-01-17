"""
Synthetic Time Series Data Generation for Clustering Evaluation

This script creates synthetic time series datasets with known cluster labels
to evaluate the performance of CRLI and VaDER clustering methods.

The synthetic data includes:
- Univariate time series: Single-dimensional signals with 3 distinct patterns
- Multivariate time series: 3-dimensional signals with feature correlations

Each dataset has 300 samples divided equally among 3 clusters with
characteristic temporal patterns that should be distinguishable by
the clustering algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory for figures
os.makedirs('figures', exist_ok=True)


def generate_red_noise(n_steps, mean=0, std=1.0, correlation=0.5):
    """
    Generate red (correlated) noise using an AR(1) process.

    Red noise, also known as Brownian noise, has temporal correlation where
    each value depends on the previous value. This creates smoother, more
    realistic noise compared to white (uncorrelated) noise.

    Parameters
    ----------
    n_steps : int
        Number of time steps to generate
    mean : float, default=0
        Mean of the noise process
    std : float, default=1.0
        Standard deviation of the noise
    correlation : float, default=0.5
        Temporal correlation coefficient (0 = white noise, 1 = random walk)

    Returns
    -------
    noise : numpy.ndarray
        Array of shape (n_steps,) containing correlated noise
    """
    noise = np.zeros(n_steps)
    noise[0] = np.random.normal(mean, std)
    for i in range(1, n_steps):
        noise[i] = correlation * noise[i-1] + np.random.normal(mean, std * np.sqrt(1 - correlation**2))
    return noise


def generate_univariate_data(n_samples=300, n_steps=100, n_clusters=3):
    """
    Generate univariate time series data with clear cluster patterns.

    Creates three types of time series patterns that should be distinguishable
    by clustering algorithms:
    - Cluster 0: Linear upward trend with low noise (smooth increase)
    - Cluster 1: Sinusoidal pattern with medium noise (oscillations)
    - Cluster 2: Linear downward trend with high noise (noisy decrease)

    Parameters
    ----------
    n_samples : int, default=300
        Total number of time series to generate (split equally among clusters)
    n_steps : int, default=100
        Length of each time series
    n_clusters : int, default=3
        Number of clusters (must be 3 for the defined patterns)

    Returns
    -------
    data : numpy.ndarray
        Array of shape (n_samples, n_steps, 1) containing time series
    labels : numpy.ndarray
        Array of shape (n_samples,) containing cluster labels (0, 1, or 2)
    """
    print(f"\nGenerating {n_samples} univariate time series with {n_steps} steps...")

    data = []
    labels = []
    samples_per_cluster = n_samples // n_clusters
    t = np.arange(n_steps)

    for cluster_id in range(n_clusters):
        for i in range(samples_per_cluster):
            if cluster_id == 0:
                # Upward trend with low noise
                base_value = np.random.uniform(5, 10)
                trend = base_value + 0.5 * t / n_steps * 10
                noise = generate_red_noise(n_steps, mean=0, std=0.5, correlation=0.5)
                ts = trend + noise

            elif cluster_id == 1:
                # Sinusoidal pattern with medium noise
                amplitude = 5
                period = 20
                phase_shift = np.random.uniform(0, 2*np.pi)
                ts = amplitude * np.sin(2 * np.pi * t / period + phase_shift) + 10
                noise = generate_red_noise(n_steps, mean=0, std=1.0, correlation=0.3)
                ts = ts + noise

            else:  # cluster_id == 2
                # Downward trend with high noise
                base_value = np.random.uniform(15, 20)
                trend = base_value - 0.5 * t / n_steps * 10
                noise = generate_red_noise(n_steps, mean=0, std=1.5, correlation=0.7)
                ts = trend + noise

            data.append(ts.reshape(n_steps, 1))  # Shape: (n_steps, 1)
            labels.append(cluster_id)

    data = np.array(data)  # Shape: (n_samples, n_steps, 1)
    labels = np.array(labels)

    print(f"Generated data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Cluster distribution: {np.bincount(labels)}")

    return data, labels


def generate_multivariate_data(n_samples=300, n_steps=100, n_features=3, n_clusters=3):
    """
    Generate multivariate time series data with clear cluster patterns.

    Creates three types of multivariate patterns where the relationships
    between features define each cluster:
    - Cluster 0: All features trend upward together (positive correlation)
    - Cluster 1: Features oscillate with phase shifts (periodic patterns)
    - Cluster 2: Features have opposite trends (anti-correlation)

    These patterns test whether clustering methods can identify relationships
    between multiple dimensions, which is a key challenge in multivariate
    time series clustering.

    Parameters
    ----------
    n_samples : int, default=300
        Total number of time series to generate (split equally among clusters)
    n_steps : int, default=100
        Length of each time series
    n_features : int, default=3
        Number of features (dimensions) per time step
    n_clusters : int, default=3
        Number of clusters (must be 3 for the defined patterns)

    Returns
    -------
    data : numpy.ndarray
        Array of shape (n_samples, n_steps, n_features) containing time series
    labels : numpy.ndarray
        Array of shape (n_samples,) containing cluster labels (0, 1, or 2)
    """
    print(f"\nGenerating {n_samples} multivariate time series with {n_steps} steps and {n_features} features...")

    data = []
    labels = []
    samples_per_cluster = n_samples // n_clusters
    t = np.arange(n_steps)

    for cluster_id in range(n_clusters):
        for i in range(samples_per_cluster):
            ts_features = []

            if cluster_id == 0:
                # Upward trends with positive correlation
                base_value = 10
                base = base_value + 0.5 * t / n_steps * 10
                for f in range(n_features):
                    noise = generate_red_noise(n_steps, mean=0, std=0.5, correlation=0.5)
                    feature = base + noise + f * 2
                    ts_features.append(feature)

            elif cluster_id == 1:
                # Sinusoidal patterns with phase shifts
                for f in range(n_features):
                    phase = f * 2 * np.pi / n_features
                    amplitude = 5
                    period = 20
                    ts = amplitude * np.sin(2 * np.pi * t / period + phase) + 10
                    noise = generate_red_noise(n_steps, mean=0, std=0.8, correlation=0.3)
                    feature = ts + noise
                    ts_features.append(feature)

            else:  # cluster_id == 2
                # Downward trends with anti-correlation
                base_value = 20
                base = base_value - 0.5 * t / n_steps * 10
                for f in range(n_features):
                    noise = generate_red_noise(n_steps, mean=0, std=1.0, correlation=0.6)
                    # Alternate direction for anti-correlation
                    sign = 1 if f % 2 == 0 else -1
                    feature = base * sign + noise
                    ts_features.append(feature)

            ts_array = np.column_stack(ts_features)  # Shape: (n_steps, n_features)
            data.append(ts_array)
            labels.append(cluster_id)

    data = np.array(data)  # Shape: (n_samples, n_steps, n_features)
    labels = np.array(labels)

    print(f"Generated data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Cluster distribution: {np.bincount(labels)}")

    return data, labels


def visualize_univariate_samples(data, labels, n_samples_per_cluster=5):
    """
    Visualize sample time series from each cluster (univariate).

    Creates a multi-panel plot showing representative examples from each
    cluster to illustrate the characteristic patterns that define each group.

    Parameters
    ----------
    data : numpy.ndarray
        Time series data of shape (n_samples, n_steps, 1)
    labels : numpy.ndarray
        Cluster labels of shape (n_samples,)
    n_samples_per_cluster : int, default=5
        Number of random samples to show per cluster

    Saves
    -----
    figures/univariate_samples.png : PNG image file
    """
    print("\nVisualizing univariate samples...")

    n_clusters = len(np.unique(labels))
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 8))

    if n_clusters == 1:
        axes = [axes]

    colors = sns.color_palette('husl', n_clusters)

    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        cluster_indices = np.where(labels == cluster_id)[0]
        sample_indices = np.random.choice(cluster_indices,
                                        min(n_samples_per_cluster, len(cluster_indices)),
                                        replace=False)

        for idx in sample_indices:
            ax.plot(data[idx, :, 0], alpha=0.7, color=colors[cluster_id], linewidth=1.5)

        ax.set_title(f'Cluster {cluster_id} (n={len(cluster_indices)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/univariate_samples.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/univariate_samples.png")
    plt.close()


def visualize_multivariate_samples(data, labels, n_samples_per_cluster=3):
    """
    Visualize sample time series from each cluster (multivariate).

    Creates a grid plot where rows represent clusters and columns represent
    features, showing how all features evolve together within each cluster.

    Parameters
    ----------
    data : numpy.ndarray
        Time series data of shape (n_samples, n_steps, n_features)
    labels : numpy.ndarray
        Cluster labels of shape (n_samples,)
    n_samples_per_cluster : int, default=3
        Number of random samples to show per cluster

    Saves
    -----
    figures/multivariate_samples.png : PNG image file
    """
    print("\nVisualizing multivariate samples...")

    n_clusters = len(np.unique(labels))
    n_features = data.shape[2]

    fig, axes = plt.subplots(n_clusters, n_features, figsize=(15, 8))

    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    colors = sns.color_palette('husl', n_clusters)

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        sample_indices = np.random.choice(cluster_indices,
                                        min(n_samples_per_cluster, len(cluster_indices)),
                                        replace=False)

        for feature_id in range(n_features):
            ax = axes[cluster_id, feature_id]

            for idx in sample_indices:
                ax.plot(data[idx, :, feature_id], alpha=0.7, color=colors[cluster_id], linewidth=1.5)

            if cluster_id == 0:
                ax.set_title(f'Feature {feature_id}', fontsize=10, fontweight='bold')
            if feature_id == 0:
                ax.set_ylabel(f'Cluster {cluster_id}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/multivariate_samples.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/multivariate_samples.png")
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION FOR CLUSTERING METHODS")
    print("=" * 60)

    # Generate univariate data
    univariate_data, univariate_labels = generate_univariate_data(
        n_samples=300,
        n_steps=100,
        n_clusters=3
    )

    # Generate multivariate data
    multivariate_data, multivariate_labels = generate_multivariate_data(
        n_samples=300,
        n_steps=100,
        n_features=3,
        n_clusters=3
    )

    # Visualize samples
    visualize_univariate_samples(univariate_data, univariate_labels)
    visualize_multivariate_samples(multivariate_data, multivariate_labels)

    # Save datasets
    print("\nSaving datasets...")
    np.savez('data_univariate.npz',
             X=univariate_data,
             y=univariate_labels)
    print("Saved: data_univariate.npz")

    np.savez('data_multivariate.npz',
             X=multivariate_data,
             y=multivariate_labels)
    print("Saved: data_multivariate.npz")

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
