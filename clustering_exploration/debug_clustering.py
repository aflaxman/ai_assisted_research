"""
Debug clustering methods to understand their return format
"""

import numpy as np
from pypots.clustering import CRLI, VaDER
import warnings
warnings.filterwarnings('ignore')

# Load univariate data
data = np.load('data_univariate.npz')
X = data['X']
y_true = data['y']

print(f"Data shape: {X.shape}")
print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
print(f"Data mean: {X.mean():.2f}, std: {X.std():.2f}")

# Normalize data
X_norm = (X - X.mean()) / X.std()
print(f"\nNormalized data range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")
print(f"Normalized data mean: {X_norm.mean():.2f}, std: {X_norm.std():.2f}")

# Test with small subset
X_test = X_norm[:30]  # Use only 30 samples for quick test
y_test = y_true[:30]

n_samples, n_steps, n_features = X_test.shape
n_clusters = len(np.unique(y_test))

print(f"\nTest data shape: {X_test.shape}")
print(f"Test clusters: {n_clusters}")

# Test CRLI
print("\n" + "="*60)
print("Testing CRLI")
print("="*60)

try:
    model = CRLI(
        n_steps=n_steps,
        n_features=n_features,
        n_clusters=n_clusters,
        n_generator_layers=1,
        rnn_hidden_size=64,
        batch_size=10,
        epochs=5,
        verbose=False
    )

    dataset = {'X': X_test}
    model.fit(dataset)

    print("\nCalling cluster()...")
    result = model.cluster(dataset)

    print(f"Result type: {type(result)}")
    if isinstance(result, dict):
        print(f"Result keys: {result.keys()}")
        for key, value in result.items():
            print(f"  {key}: type={type(value)}, ", end='')
            if isinstance(value, np.ndarray):
                print(f"shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"value={value}")
    else:
        print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        print(f"Result dtype: {result.dtype if hasattr(result, 'dtype') else 'N/A'}")
        print(f"Result sample: {result[:5] if hasattr(result, '__getitem__') else result}")

except Exception as e:
    print(f"CRLI error: {e}")
    import traceback
    traceback.print_exc()

# Test VaDER
print("\n" + "="*60)
print("Testing VaDER")
print("="*60)

try:
    model = VaDER(
        n_steps=n_steps,
        n_features=n_features,
        n_clusters=n_clusters,
        rnn_hidden_size=64,
        d_mu_stddev=5,
        batch_size=10,
        epochs=5,
        pretrain_epochs=2,
        verbose=False
    )

    dataset = {'X': X_test}
    model.fit(dataset)

    print("\nCalling cluster()...")
    result = model.cluster(dataset)

    print(f"Result type: {type(result)}")
    if isinstance(result, dict):
        print(f"Result keys: {result.keys()}")
        for key, value in result.items():
            print(f"  {key}: type={type(value)}, ", end='')
            if isinstance(value, np.ndarray):
                print(f"shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"value={value}")
    else:
        print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        print(f"Result dtype: {result.dtype if hasattr(result, 'dtype') else 'N/A'}")
        print(f"Result sample: {result[:5] if hasattr(result, '__getitem__') else result}")

except Exception as e:
    print(f"VaDER error: {e}")
    import traceback
    traceback.print_exc()
