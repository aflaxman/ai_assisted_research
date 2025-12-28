"""
CRLI vs VaDER Clustering Evaluation

This script evaluates and compares two deep learning time series clustering
methods from the PyPOTS library:

1. CRLI: GAN-based clustering with k-means guidance
2. VaDER: VAE-based clustering with variational inference

The evaluation includes:
- Training both methods on univariate and multivariate time series
- Computing clustering accuracy metrics (ARI, NMI)
- Measuring training time
- Visualizing results using t-SNE projections

Results are saved to figures/ directory and logged to console.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score
)
from sklearn.manifold import TSNE
from pypots.clustering import CRLI, VaDER
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

def load_data(data_type='univariate'):
    """
    Load and preprocess synthetic time series data for clustering.

    Loads the .npz data files created by generate_data.py and applies
    z-score normalization, which is essential for neural network stability.

    Parameters
    ----------
    data_type : str, default='univariate'
        Type of data to load: 'univariate' or 'multivariate'

    Returns
    -------
    X : numpy.ndarray
        Normalized time series data
    y_true : numpy.ndarray
        Ground truth cluster labels
    """
    if data_type == 'univariate':
        data = np.load('data_univariate.npz')
    else:
        data = np.load('data_multivariate.npz')

    X = data['X']
    y_true = data['y']

    # Normalize data (important for neural networks)
    X = (X - X.mean()) / X.std()

    print(f"\nLoaded {data_type} data:")
    print(f"  Shape: {X.shape}")
    print(f"  True clusters: {len(np.unique(y_true))}")
    print(f"  Cluster distribution: {np.bincount(y_true)}")
    print(f"  Data range: [{X.min():.2f}, {X.max():.2f}]")

    return X, y_true


def prepare_pypots_data(X):
    """
    Prepare data for PyPOTS models.
    PyPOTS expects data in the format: dict with 'X' key
    """
    # Create dataset dictionary
    dataset = {'X': X}
    return dataset


def compute_metrics(y_true, y_pred, X_embedded=None):
    """
    Compute clustering evaluation metrics.

    Calculates two primary metrics for clustering quality:
    - ARI (Adjusted Rand Index): Measures similarity between predicted and
      true clusters, adjusted for chance. Range: [-1, 1], higher is better.
    - NMI (Normalized Mutual Information): Measures shared information
      between predicted and true clusters. Range: [0, 1], higher is better.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth cluster labels
    y_pred : numpy.ndarray
        Predicted cluster labels
    X_embedded : numpy.ndarray, optional
        Embedded feature vectors (not used in current implementation)

    Returns
    -------
    metrics : dict
        Dictionary containing 'ARI' and 'NMI' scores
    """
    metrics = {}

    # Adjusted Rand Index (1.0 is perfect, 0.0 is random)
    metrics['ARI'] = adjusted_rand_score(y_true, y_pred)

    # Normalized Mutual Information (1.0 is perfect, 0.0 is independent)
    metrics['NMI'] = normalized_mutual_info_score(y_true, y_pred)

    # Silhouette Score (uses embedded features if available)
    if X_embedded is not None:
        try:
            metrics['Silhouette'] = silhouette_score(X_embedded, y_pred)
        except:
            metrics['Silhouette'] = np.nan
    else:
        metrics['Silhouette'] = np.nan

    # Davies-Bouldin Index (lower is better, uses embedded features)
    if X_embedded is not None:
        try:
            metrics['Davies-Bouldin'] = davies_bouldin_score(X_embedded, y_pred)
        except:
            metrics['Davies-Bouldin'] = np.nan
    else:
        metrics['Davies-Bouldin'] = np.nan

    return metrics


def test_crli(X, y_true, data_type='univariate', epochs=30):
    """Test CRLI clustering method."""
    print(f"\n{'='*60}")
    print(f"Testing CRLI on {data_type} data")
    print(f"{'='*60}")

    n_samples, n_steps, n_features = X.shape
    n_clusters = len(np.unique(y_true))

    # Create CRLI model
    print("\nInitializing CRLI model...")
    model = CRLI(
        n_steps=n_steps,
        n_features=n_features,
        n_clusters=n_clusters,
        n_generator_layers=2,
        rnn_hidden_size=128,
        rnn_cell_type='GRU',
        lambda_kmeans=1.0,
        batch_size=32,
        epochs=epochs,
        patience=10,
        verbose=False
    )

    # Prepare data
    dataset = prepare_pypots_data(X)

    # Train and cluster
    print(f"Training CRLI (epochs={epochs})...")
    start_time = time.time()

    model.fit(dataset)
    y_pred = model.cluster(dataset)  # Returns numpy array directly

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")

    # Get learned representations for visualization
    # Note: CRLI doesn't directly expose embeddings, so we'll skip detailed metrics
    metrics = compute_metrics(y_true, y_pred, X_embedded=None)

    print("\nMetrics:")
    for metric_name, metric_value in metrics.items():
        if not np.isnan(metric_value):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: N/A")

    return {
        'method': 'CRLI',
        'data_type': data_type,
        'y_pred': y_pred,
        'train_time': train_time,
        'metrics': metrics,
        'model': model
    }


def test_vader(X, y_true, data_type='univariate', epochs=30, pretrain_epochs=10):
    """Test VaDER clustering method."""
    print(f"\n{'='*60}")
    print(f"Testing VaDER on {data_type} data")
    print(f"{'='*60}")

    n_samples, n_steps, n_features = X.shape
    n_clusters = len(np.unique(y_true))

    # Create VaDER model with lower learning rate to avoid NaN
    print("\nInitializing VaDER model...")
    from pypots.optim import Adam

    optimizer = Adam(lr=0.0005)  # Lower learning rate to prevent NaN

    model = VaDER(
        n_steps=n_steps,
        n_features=n_features,
        n_clusters=n_clusters,
        rnn_hidden_size=64,  # Smaller hidden size for stability
        d_mu_stddev=5,       # Smaller dimension
        batch_size=32,
        epochs=epochs,
        pretrain_epochs=pretrain_epochs,
        patience=10,
        optimizer=optimizer,
        verbose=False
    )

    # Prepare data
    dataset = prepare_pypots_data(X)

    # Train and cluster
    print(f"Training VaDER (pretrain={pretrain_epochs}, epochs={epochs})...")
    start_time = time.time()

    model.fit(dataset)
    y_pred = model.cluster(dataset)  # Returns numpy array directly

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, X_embedded=None)

    print("\nMetrics:")
    for metric_name, metric_value in metrics.items():
        if not np.isnan(metric_value):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: N/A")

    return {
        'method': 'VaDER',
        'data_type': data_type,
        'y_pred': y_pred,
        'train_time': train_time,
        'metrics': metrics,
        'model': model
    }


def visualize_results(results_list, X, y_true, data_type):
    """Visualize clustering results."""
    print(f"\nVisualizing {data_type} results...")

    n_methods = len(results_list)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 4))

    # Flatten time series for t-SNE
    n_samples, n_steps, n_features = X.shape
    X_flat = X.reshape(n_samples, n_steps * n_features)

    # Compute t-SNE for visualization
    print("  Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_flat)

    # Plot ground truth
    ax = axes[0]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=30)
    ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax)

    # Plot each method's results
    for i, result in enumerate(results_list):
        ax = axes[i + 1]
        y_pred = result['y_pred']
        method = result['method']
        ari = result['metrics']['ARI']
        nmi = result['metrics']['NMI']
        time_taken = result['train_time']

        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, cmap='viridis', alpha=0.6, s=30)
        ax.set_title(f"{method}\nARI: {ari:.3f}, NMI: {nmi:.3f}\nTime: {time_taken:.1f}s",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.savefig(f'figures/{data_type}_clustering_results.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: figures/{data_type}_clustering_results.png")
    plt.close()


def main():
    """Main function to run all tests."""
    print("="*60)
    print("CLUSTERING METHODS COMPARISON")
    print("="*60)

    all_results = []

    # Test on univariate data
    print("\n" + "#"*60)
    print("# UNIVARIATE DATA")
    print("#"*60)

    X_uni, y_uni = load_data('univariate')

    crli_uni = test_crli(X_uni, y_uni, 'univariate', epochs=30)
    all_results.append(crli_uni)

    vader_uni = test_vader(X_uni, y_uni, 'univariate', epochs=30, pretrain_epochs=10)
    all_results.append(vader_uni)

    visualize_results([crli_uni, vader_uni], X_uni, y_uni, 'univariate')

    # Test on multivariate data
    print("\n" + "#"*60)
    print("# MULTIVARIATE DATA")
    print("#"*60)

    X_multi, y_multi = load_data('multivariate')

    crli_multi = test_crli(X_multi, y_multi, 'multivariate', epochs=30)
    all_results.append(crli_multi)

    vader_multi = test_vader(X_multi, y_multi, 'multivariate', epochs=30, pretrain_epochs=10)
    all_results.append(vader_multi)

    visualize_results([crli_multi, vader_multi], X_multi, y_multi, 'multivariate')

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    print("\n{:<15} {:<15} {:<12} {:<12} {:<12}".format(
        "Method", "Data Type", "ARI", "NMI", "Time (s)"))
    print("-"*60)

    for result in all_results:
        print("{:<15} {:<15} {:<12.4f} {:<12.4f} {:<12.2f}".format(
            result['method'],
            result['data_type'],
            result['metrics']['ARI'],
            result['metrics']['NMI'],
            result['train_time']
        ))

    print("\n" + "="*60)
    print("TESTS COMPLETE!")
    print("="*60)

    return all_results


if __name__ == '__main__':
    results = main()
