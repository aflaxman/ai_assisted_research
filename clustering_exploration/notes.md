# Clustering Exploration Notes

## Session Log

### Initial Setup (2025-11-13)

**Environment Setup:**
- Created `clustering_exploration` folder
- Initialized uv project
- Installed dependencies:
  - pypots==1.0
  - mockseries==0.3.1
  - matplotlib==3.7.1
  - seaborn==0.13.2
  - pandas==2.3.3
  - numpy==1.26.3
  - scikit-learn==1.7.2
  - torch==2.9.1 (with CUDA support)

**Next Steps:**
- Research CRLI and VaDER clustering methods in PyPOTS
- Understand their parameters and use cases
- Design synthetic test data generation strategy

### PyPOTS Clustering Methods Research

**CRLI (Clustering Representation Learning on Incomplete time-series data)**
- Paper citation: ma2021CRLI
- Architecture: GAN-based (Generator + Discriminator)
- Key parameters:
  - n_generator_layers: Number of generator layers
  - rnn_hidden_size: Size of RNN hidden state
  - rnn_cell_type: 'GRU' or 'LSTM'
  - lambda_kmeans: Weight of k-means loss
  - G_steps, D_steps: Training steps for generator/discriminator
- Use case: Designed for incomplete/missing time-series data
- Training: Alternates between generator and discriminator training

**VaDER (Variational Deep Embedding with Recurrence)**
- Paper citation: dejong2019VaDER
- Architecture: Variational Autoencoder (VAE)
- Key parameters:
  - rnn_hidden_size: Size of RNN hidden state
  - d_mu_stddev: Dimension of Gaussian distribution parameters
  - pretrain_epochs: Epochs for pretraining before main training
- Use case: Deep clustering with variational inference
- Training: Has pretrain phase followed by main training

**Common Parameters:**
- n_steps: Length of time series
- n_features: Number of features (dimensions)
- n_clusters: Number of clusters to find
- batch_size: Training batch size
- epochs: Training epochs
- patience: Early stopping patience

### Data Generation Exploration

**mockseries API:**
- Requires datetime.timedelta for time_unit and period
- Not simple numeric time steps
- More complex than needed for this task

**Decision:** Use numpy directly for more control over synthetic data generation
- Easier to create specific cluster patterns
- More flexible for testing purposes
- Can create controlled scenarios

### Initial Clustering Tests

**Issues encountered:**
1. CRLI produced NaN values initially
   - Fixed by: data normalization and correct return format handling
   - cluster() returns numpy array, not dict

2. VaDER producing NaN values during training
   - Error in pretrain phase: `predictions mustn't contain NaN values`
   - Likely due to network instability
   - Need to adjust learning rate or initialization

**CRLI Results (Univariate, 30 epochs):**
- Training time: 378.72 seconds (~6.3 minutes)
- ARI: 0.4974
- NMI: 0.5526
- Status: ✓ Working but moderate accuracy

**Action:** Adjust VaDER parameters - try lower learning rate and different hidden size

### Final Test Results

**Univariate Time Series (300 samples, 100 steps, 1 feature, 3 clusters):**

CRLI:
- Training time: 395.92 seconds (~6.6 minutes)
- ARI: 0.5179
- NMI: 0.5679
- Status: ✓ Good accuracy, slower training

VaDER:
- Training time: 56.80 seconds (~0.95 minutes)
- ARI: 0.3515
- NMI: 0.4922
- Status: ✓ Fast training, moderate accuracy

**Multivariate Time Series (300 samples, 100 steps, 3 features, 3 clusters):**

CRLI:
- Training time: 424.34 seconds (~7.1 minutes)
- ARI: 1.0000 (PERFECT!)
- NMI: 1.0000 (PERFECT!)
- Status: ✓ Excellent accuracy, slower training

VaDER:
- Training time: 88.27 seconds (~1.5 minutes)
- ARI: 0.1271
- NMI: 0.2834
- Status: ✓ Fast training, lower accuracy

### Key Findings

1. **CRLI excels at accuracy** - especially on multivariate data where it achieved perfect clustering
2. **VaDER excels at speed** - approximately 7x faster than CRLI
3. **Multivariate benefits CRLI** - more features provide richer signal for CRLI's GAN architecture
4. **Parameter tuning critical** - VaDER required lower learning rate (0.0005) to avoid NaN issues
5. **Trade-off** - Speed vs. Accuracy: Choose VaDER for fast exploratory analysis, CRLI for final results

### Technical Notes

- Data normalization essential for both methods
- CRLI: GAN-based approach with 2 generator layers, 128 hidden units
- VaDER: VAE-based approach with 64 hidden units, 5 latent dimensions
- Both methods use batch size 32, 30 epochs, patience 10

---

