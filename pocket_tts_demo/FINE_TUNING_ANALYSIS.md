# Fine-Tuning Pocket TTS - Technical Analysis

## Executive Summary

**You're absolutely right** - fine-tuning Pocket TTS IS possible! While Kyutai hasn't released training code yet, we have:
- ✅ Complete model weights
- ✅ Full architecture details in the code
- ✅ Published training methodology in the paper
- ✅ Access to the model internals via the installed package

## What We Know About Training

### Architecture Components

From the paper ([Continuous Audio Language Models, arXiv:2509.06926](https://arxiv.org/html/2509.06926)):

1. **Mimi VAE Encoder/Decoder**
   - Converts audio ↔ 32-dim continuous latents
   - 12.5 Hz frame rate (80ms per frame)
   - Already frozen during TTS training

2. **Transformer Backbone**
   - StreamingTransformer (24 layers for full model)
   - Pocket TTS uses 6 layers (distilled from 24-layer teacher)
   - Causal attention with positional embeddings

3. **Flow Matching Head (MLP)**
   - SimpleMLPAdaLN - MLP with adaptive layer normalization
   - Predicts velocity field for continuous latent diffusion
   - Conditioned on text embeddings and time

4. **Text Conditioner**
   - SentencePiece tokenizer (4k vocabulary)
   - LUT (lookup table) embeddings

### Training Methodology

**Loss Function:**
```
Consistency loss with flow matching:
L = E[w(t) || flow_net(x_t, t, cond) - v_t ||²]

where:
- x_t = noisy latent at time t
- v_t = target velocity field
- w(t) = adaptive weighting function
- cond = text embeddings
```

**Key Training Details:**
- 75% flow matching loss + 25% LSD (Lagrangian Self-Distillation) loss
- Gaussian noise schedule: α_t = cos(πt/2), σ_t = sin(πt/2)
- Temperature: 0.8 for speech
- CFG coefficient: α = 1.5 for distillation

### Dataset Requirements

**For TTS Training (from paper):**
- 88k hours of speech data
- Mix of: LibriSpeech, LibriHeavy, GigaSpeech, TED-LIUM 3, Earnings-22, AMI
- Requires paired (audio, text) transcripts
- Preprocessed to 32-dim Mimi latents at 12.5 Hz

**For Fine-Tuning (realistic):**
- Much smaller dataset feasible (hundreds of hours, not thousands)
- High-quality audio + accurate transcripts
- Specific domain/style you want to emphasize

## What's Needed to Implement Fine-Tuning

### 1. Access Model Internals ✅

The installed package already exposes everything:

```python
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.models.flow_lm import FlowLMModel
import torch

# Load the model
tts = TTSModel.load_model()

# Access trainable components
flow_lm = tts.flow_lm  # The transformer + flow head
conditioner = flow_lm.conditioner  # Text embeddings
flow_net = flow_lm.flow_net  # MLP head
transformer = flow_lm.transformer  # Main transformer

# All weights are accessible
for name, param in flow_lm.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
```

### 2. Implement Training Loop

**What's missing:**
- Training script to load data
- Loss computation functions
- Optimizer setup
- Training loop

**What we can build:**

```python
# Pseudocode for fine-tuning loop

optimizer = torch.optim.AdamW(flow_lm.parameters(), lr=1e-4)

for batch in dataloader:
    audio, text = batch

    # 1. Extract Mimi latents (encoder is frozen)
    with torch.no_grad():
        latents = tts.mimi.encode(audio)  # (B, T, 32)

    # 2. Get text embeddings
    text_emb = flow_lm.conditioner(text)  # (B, T_text, dim)

    # 3. Sample time and add noise (flow matching)
    t = torch.rand(B, 1)
    noise = torch.randn_like(latents)
    x_t = sqrt(1-t) * latents + sqrt(t) * noise

    # 4. Predict velocity
    v_pred = flow_lm(x_t, text_emb, t)

    # 5. Compute loss (target is normalized difference)
    v_target = noise - latents
    loss = F.mse_loss(v_pred, v_target)

    # 6. Backprop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 3. Dataset Preparation

**Option A: Use existing TTS datasets**
- LJSpeech (24 hours, single speaker)
- LibriTTS (585 hours, multi-speaker)
- Common Voice (thousands of hours)

**Option B: Create custom dataset**
- Record your own audio with transcripts
- Use existing audio + Whisper for transcription
- Focus on your target voice style/domain

**Preprocessing needed:**
```python
# Convert audio to Mimi latents
from pocket_tts.models.mimi import Mimi

mimi = Mimi.load_model()
latents = mimi.encode(audio_waveform)  # (T, 32)

# Tokenize text
from pocket_tts.conditioners.text import LUTConditioner

conditioner = LUTConditioner(...)
text_tokens = conditioner.tokenize(transcript)
```

### 4. Compute Requirements

**Estimates based on model size:**

**Pocket TTS (100M params):**
- GPU: 1x RTX 4090 (24GB) or A100 (40GB)
- Batch size: 8-16 depending on sequence length
- Training time: Days to weeks depending on dataset size
- Fine-tuning: Faster, possibly 1-3 days for small datasets

**Full model (2B params):**
- GPU: 4-8x A100s or H100s
- Much more expensive, probably not needed for fine-tuning

## Practical Fine-Tuning Approaches

### Approach 1: Fine-tune MLP Head Only (Easiest)

Freeze the transformer, only train the flow_net MLP:

```python
# Freeze transformer
for param in flow_lm.transformer.parameters():
    param.requires_grad = False

# Only train the MLP head
optimizer = torch.optim.AdamW(flow_lm.flow_net.parameters(), lr=1e-4)
```

**Advantages:**
- Much faster training
- Lower GPU memory requirements
- Less risk of catastrophic forgetting
- Good for adapting to new speaking styles

### Approach 2: Fine-tune Last N Layers (Medium)

Freeze early layers, fine-tune last layers + head:

```python
# Freeze first 4 layers (out of 6)
for i in range(4):
    for param in flow_lm.transformer.layers[i].parameters():
        param.requires_grad = False

# Train last 2 layers + MLP
trainable = list(flow_lm.transformer.layers[4:].parameters()) + \
            list(flow_lm.flow_net.parameters())
optimizer = torch.optim.AdamW(trainable, lr=5e-5)
```

### Approach 3: Full Fine-Tuning (Hardest)

Train everything (except frozen Mimi):

```python
optimizer = torch.optim.AdamW(flow_lm.parameters(), lr=1e-5)
```

**Warning:** Requires careful hyperparameter tuning and more data.

## Why Kyutai Says "Training Not Available"

They mean:
1. **No training scripts** - You have to write them yourself
2. **No training documentation** - You need to read the paper
3. **No official support** - It's research, not a product

But the model architecture and weights are fully accessible!

## What Training Code Would Look Like

Here's what's needed (rough estimate):

1. **Data loading pipeline** (~500 lines)
   - Audio loading and preprocessing
   - Mimi latent extraction
   - Text tokenization
   - Batching with proper padding

2. **Training loop** (~200 lines)
   - Flow matching loss implementation
   - LSD loss (optional, for advanced users)
   - Optimizer setup with learning rate scheduling
   - Checkpointing and logging

3. **Evaluation** (~100 lines)
   - Generate samples during training
   - Compute metrics (if available)
   - Listen to outputs

**Total: ~800-1000 lines of code**

This is totally feasible to implement!

## Recommended Next Steps

### Step 1: Verify Model Access
```bash
cd pocket_tts_demo
python verify_model_access.py
```

### Step 2: Start with Minimal Training Script
Create a proof-of-concept that:
- Loads one audio file
- Extracts latents
- Runs one forward pass
- Computes loss
- Backprops once

### Step 3: Scale Up
Once POC works:
- Add data loading
- Add checkpointing
- Train on small dataset (LJSpeech)
- Evaluate results

### Step 4: Fine-tune for Your Use Case
- Gritty voice? Use gritty audio in training data
- Specific domain? Use domain-specific text/audio
- New language? Collect new language data

## Key Insight

The paper says they "will release training code in order to retrain CALM" - this was conditional on publication/acceptance. Since the paper is published and the model is released, **the architecture is public knowledge**.

You have everything needed to implement training yourself:
- ✅ Model architecture (in the installed package)
- ✅ Training algorithm (in the paper)
- ✅ Pre-trained weights (to fine-tune from)
- ✅ Example data processing (in the inference code)

**Bottom Line: Fine-tuning is absolutely possible, you just need to write the training script!**

## Resources

- [CALM Paper (arXiv:2509.06926)](https://arxiv.org/html/2509.06926)
- [Pocket TTS GitHub](https://github.com/kyutai-labs/pocket-tts)
- [Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- Model code: `/usr/local/lib/python3.11/dist-packages/pocket_tts/`

## References

All training details extracted from:
- Rouard, S., Orsini, M., Roebel, A., Zeghidour, N., & Défossez, A. (2025). Continuous Audio Language Models. arXiv:2509.06926v3.
