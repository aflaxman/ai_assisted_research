# Comprehensive Fine-Tuning Research: Pocket TTS vs Resemble-Enhance vs Chatterbox TTS

## Executive Summary: Important Clarification

**You're absolutely right** - fine-tuning IS possible! This analysis clarifies three distinct projects:

1. **Pocket TTS** (Kyutai Labs) - 100M parameter TTS model (CPU-optimized) - **What you're currently using**
2. **Resemble-Enhance** (Resemble AI) - Audio enhancement tool (denoising/enhancement, NOT TTS)
3. **Chatterbox TTS** (Resemble AI) - Production TTS with extensive fine-tuning support

**Bottom Line:** We have the weights, we have the architecture, and fine-tuning is absolutely possible for all three. The difference is in the tooling available.

---

## 1. POCKET TTS (Kyutai Labs) - What You're Currently Using

### Model Architecture

**Based on: Continuous Audio Language Models (CALM)**

From the [arXiv paper 2509.06926](https://arxiv.org/abs/2509.06926):

**Core Components:**
- **Mimi VAE** (32-dim continuous latents at 12.5 Hz)
- **Transformer Backbone** (300M params for base, distilled to 100M for Pocket)
- **Flow Matching Head** (MLP with adaptive layer normalization)
- **Text Conditioner** (SentencePiece tokenizer, 4k vocabulary)

**Architecture Innovation:**
- Uses **continuous audio modeling** instead of discrete tokens
- Avoids lossy compression of traditional audio codecs
- "Inner monologue" mechanism - latent textual representation aligned with audio timesteps

### Fine-Tuning Feasibility: ✅ YES, ABSOLUTELY POSSIBLE

**What you already have:**
- ✅ Complete model weights (PyTorch format)
- ✅ Full architecture accessible via Python API
- ✅ All parameters are trainable (`requires_grad=True`)
- ✅ Published training methodology in paper
- ✅ Gradient computation works (verified in `verify_model_access.py`)

**What's missing:**
- ⚠️ Official training scripts (you need to write ~800 lines of code)
- ⚠️ Dataset preprocessing pipeline
- ⚠️ Loss function implementation (flow matching + LSD)

### Training Methodology

**Loss Function:**
```python
# Combined loss
L = 0.75 * flow_matching_loss + 0.25 * LSD_loss

# Flow matching: predict velocity field
v_target = (noise - latents) / sqrt(1 - t)
v_pred = flow_net(x_t, text_emb, t)
flow_loss = MSE(v_pred, v_target)

# Noise schedule
alpha_t = cos(π*t/2)  # signal weight
sigma_t = sin(π*t/2)  # noise weight
```

**Dataset Requirements:**
- Original training: 88,000 hours (LibriSpeech, GigaSpeech, TED-LIUM, etc.)
- For fine-tuning: 100-1000 hours would be sufficient
- Format: Paired (audio, text) with accurate transcripts

### Practical Fine-Tuning Approaches

#### Option 1: MLP Head Only (Easiest)
```python
# Freeze transformer, train only flow_net
for param in tts.flow_lm.transformer.parameters():
    param.requires_grad = False
optimizer = torch.optim.AdamW(tts.flow_lm.flow_net.parameters(), lr=1e-4)
```
- **GPU:** 1x RTX 4090 (24GB)
- **Time:** 1-3 days for small dataset
- **Best for:** Adapting to new speaking styles

#### Option 2: Last N Layers (Medium)
```python
# Freeze first 4 layers, train last 2 + MLP
for i in range(4):
    for param in tts.flow_lm.transformer.layers[i].parameters():
        param.requires_grad = False
```
- **GPU:** 1x A100 (40GB recommended)
- **Time:** 3-7 days
- **Best for:** Domain adaptation

#### Option 3: Full Fine-Tuning (Advanced)
```python
optimizer = torch.optim.AdamW(tts.flow_lm.parameters(), lr=1e-5)
```
- **GPU:** 2-4x A100
- **Time:** 1-2 weeks
- **Best for:** New languages or significant distribution shift

### Model Weights Format
- **Format:** PyTorch `.pt` or `.safetensors`
- **Location:** Downloaded from Hugging Face Hub (`hf://kyutai/pocket-tts`)
- **Size:** ~400MB (100M parameters × 4 bytes/float32)
- **Accessibility:** Full read/write access via Python API

### Accessing Model Internals

```python
from pocket_tts.models.tts_model import TTSModel
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

### Pseudocode for Training Loop

```python
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

---

## 2. RESEMBLE-ENHANCE (Resemble AI) - Audio Enhancement, NOT TTS

### Model Architecture

**Purpose:** Audio post-processing (denoising + enhancement), NOT text-to-speech

**Components:**
1. **Denoiser Module** - Separates speech from noise
2. **Enhancer Module** - Two-stage training:
   - Stage 1: Autoencoder + Vocoder
   - Stage 2: Latent Conditional Flow Matching (CFM)

**Training Data:** 44.1kHz high-quality speech

### Fine-Tuning Feasibility: ✅ YES, FULL TRAINING SCRIPTS AVAILABLE

**Official Training Scripts:**
```bash
# Step 1: Train denoiser (warmup)
python -m resemble_enhance.denoiser.train --yaml config/denoiser.yaml runs/denoiser

# Step 2: Train enhancer stage 1 (autoencoder + vocoder)
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage1.yaml runs/enhancer_stage1

# Step 3: Train enhancer stage 2 (flow matching)
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage2.yaml runs/enhancer_stage2
```

**Dataset Requirements:**
- **Foreground (fg):** Clean speech WAV files
- **Background (bg):** Non-speech noise audio
- **RIR:** Room impulse response files (`.npy` format)

**Key Advantage:** Unlike Pocket TTS, Resemble-Enhance provides **complete training code out-of-the-box**.

### Installation
```bash
pip install resemble-enhance --upgrade
```

**Use Case:** If you want to enhance/denoise audio AFTER generating it with TTS, not for TTS generation itself.

---

## 3. CHATTERBOX TTS (Resemble AI) - Actual TTS with Fine-Tuning

### Model Architecture

**Overview:** State-of-the-art open-source TTS model, benchmarked against ElevenLabs

**Variants:**
- **Chatterbox TTS** (standard)
- **Chatterbox Turbo** (faster inference)
- **Chatterbox Multilingual** (23 languages)

**License:** MIT

### Fine-Tuning Feasibility: ✅ YES, WITH DEDICATED TOOLKIT

**Simple Fine-Tuning Process:**
```bash
# 1. Prepare data: Place WAV files in audio_data/
mkdir audio_data
cp your_voice_samples/*.wav audio_data/

# 2. Run fine-tuning script
python lora.py
```

**Advanced Toolkit:** [gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning)
- Supports 23 languages
- Smart vocabulary extension
- Automatic VAD trimming
- Voice cloning capabilities
- LJSpeech and file-based formats

**Training Requirements:**
- **GPU:** CUDA with 18GB+ VRAM
- **Data:** 1 hour of target speaker audio
- **Training:** 150 epochs or 1000 steps
- **Time:** Hours to days depending on GPU

**Output Format:** `.safetensors`

**Key Advantage:** Most accessible fine-tuning process - just provide WAV files and run!

---

## 4. Technical Feasibility Comparison

| Project | Training Scripts | Weight Access | Fine-Tuning Ease | Best For |
|---------|-----------------|---------------|-------------------|----------|
| **Pocket TTS** | ❌ (DIY) | ✅ Full | ⚠️ Medium (800 LOC) | CPU inference, research |
| **Resemble-Enhance** | ✅ Complete | ✅ Full | ✅ Easy (ready scripts) | Audio enhancement |
| **Chatterbox TTS** | ✅ Toolkit | ✅ Full | ✅ Very Easy (LoRA) | Production TTS |

---

## 5. Similar TTS Models with Fine-Tuning Support

### XTTS-v2 (Coqui AI)
- **Fine-tuning:** [Official guide](https://docs.coqui.ai/en/latest/models/xtts.html)
- **Training recipe:** Available on [GitHub](https://github.com/coqui-ai/TTS/blob/dev/recipes/ljspeech/xtts_v2/train_gpt_xtts.py)
- **Gradio demo:** Pre-built fine-tuning UI
- **Requirements:** BATCH_SIZE × GRAD_ACCUM ≥ 252
- **Guides:** Simple & Advanced versions available
- **Tutorial:** [Kaggle notebook](https://www.kaggle.com/code/maxbr0wn/tutorial-fine-tuning-xttsv2-english)

### F5-TTS
- **Fine-tuning:** Active development with community support
- **Weights:** PyTorch → safetensors conversion available
- **Resume training:** Copy `F5TTS_Base/model_1200000.pt` to training folder
- **VRAM:** 12GB → batch_size=4; lower → batch_size=2, grad_accum=32
- **Discussions:** [GitHub](https://github.com/SWivid/F5-TTS/discussions/57)

### SpeechT5 (Microsoft)
- **Fine-tuning:** [Colab notebook](https://colab.research.google.com/drive/1i7I5pzBcU3WDFarDnzweIj4-sVVoIUFJ)
- **Framework:** Hugging Face Transformers
- **Ease:** Beginner-friendly with official tutorials

---

## 6. Recommended Path Forward

### For Your Use Case (Gritty/Breathy Voice)

Based on your existing project focusing on voice characteristics:

**Short-term (hours):** Use voice cloning with custom recordings
- Already implemented in `use_your_voice.py`
- Record gritty/breathy voice sample
- Model clones characteristics without training

**Medium-term (days):** Fine-tune Chatterbox TTS
- Collect 1 hour of gritty voice audio
- Use chatterbox-finetuning toolkit
- Result: Custom voice model in days

**Long-term (weeks):** Implement Pocket TTS fine-tuning
- Write training loop (~800 LOC)
- Fine-tune on gritty speech dataset
- Most control, highest effort

### Implementation Paths

#### Option A: Fine-tune Pocket TTS specifically

1. **Verify model access** (already done):
   ```bash
   cd /home/user/ai_assisted_research/pocket_tts_demo
   python verify_model_access.py
   ```

2. **Implement minimal training loop** (~800 lines):
   - Data loader for (audio, text) pairs
   - Flow matching loss function
   - Training loop with checkpointing
   - Reference: See pseudocode above

3. **Start with small dataset**:
   - Download LJSpeech (24 hours, single speaker)
   - Test MLP-only fine-tuning first

4. **Scale up**:
   - Larger datasets (LibriTTS)
   - Last-N-layers fine-tuning
   - Domain-specific data

#### Option B: Easiest fine-tuning NOW

1. **Switch to Chatterbox TTS**:
   ```bash
   pip install chatterbox-tts
   git clone https://github.com/gokhaneraslan/chatterbox-finetuning
   cd chatterbox-finetuning
   # Place WAVs in audio_data/, run lora.py
   ```

2. **Or use XTTS-v2**:
   ```bash
   pip install TTS
   # Use official Gradio fine-tuning demo
   ```

#### Option C: Audio enhancement pipeline

1. **Install Resemble-Enhance**:
   ```bash
   pip install resemble-enhance --upgrade
   ```

2. **Use for post-processing**:
   ```bash
   # Generate TTS with Pocket TTS
   # Then enhance with Resemble-Enhance
   resemble-enhance input_dir output_dir
   ```

---

## 7. Key Insights

### Why "We Have Weights, We Have Architecture" Is Correct

You're absolutely right! The key insight:

**Pocket TTS:**
- ✅ Weights: Accessible via `tts.flow_lm.state_dict()`
- ✅ Architecture: Fully documented in code
- ✅ Gradients: Computed successfully
- ✅ Training algorithm: Published in paper
- ⚠️ Training scripts: Not provided (but you can write them)

**Resemble-Enhance:**
- ✅ Weights: Provided
- ✅ Architecture: Two-module system
- ✅ Training scripts: **Complete and ready to use**
- ✅ Dataset format: Documented

**Chatterbox TTS:**
- ✅ Weights: Provided
- ✅ Architecture: Production-grade TTS
- ✅ Fine-tuning toolkit: **Community-built, ready**
- ✅ LoRA support: Efficient fine-tuning

### Dataset Options

**For TTS Training (Pocket TTS):**

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

### Compute Requirements

**Pocket TTS (100M params):**
- GPU: 1x RTX 4090 (24GB) or A100 (40GB)
- Batch size: 8-16 depending on sequence length
- Training time: Days to weeks depending on dataset size
- Fine-tuning: Faster, possibly 1-3 days for small datasets

**Full model (2B params):**
- GPU: 4-8x A100s or H100s
- Much more expensive, probably not needed for fine-tuning

---

## 8. What Training Code Would Look Like

**Total estimate: ~800-1000 lines of code**

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

This is totally feasible to implement!

---

## 9. Conclusion

**Your statement "we have the weights, we have the architecture, so we should be able to tune!" is 100% correct** for all three projects.

The difference is in the tooling:
- **Pocket TTS:** DIY training loop
- **Resemble-Enhance:** Complete training scripts
- **Chatterbox:** Fine-tuning toolkit

**For your gritty/breathy voice use case:**
1. **Quick win:** Use voice cloning with custom recordings (already implemented)
2. **Best ROI:** Fine-tune Chatterbox TTS (easiest path to custom model)
3. **Research path:** Implement Pocket TTS training (most control, highest effort)

All paths are viable - choose based on your timeline and technical goals!

---

## Resources & Sources

### Pocket TTS (Kyutai)
- [Pocket TTS GitHub](https://github.com/kyutai-labs/pocket-tts)
- [CALM Paper (arXiv:2509.06926)](https://arxiv.org/abs/2509.06926)
- [Continuous Audio Language Models (HTML)](https://arxiv.org/html/2509.06926v2)
- [Kyutai Blog Post](https://kyutai.org/blog/2026-01-13-pocket-tts)
- [HuggingFace Model](https://huggingface.co/kyutai/pocket-tts)

### Resemble-Enhance (Resemble AI)
- [GitHub Repository](https://github.com/resemble-ai/resemble-enhance)
- [README](https://github.com/resemble-ai/resemble-enhance/blob/main/README.md)
- [HuggingFace Model](https://huggingface.co/ResembleAI/resemble-enhance)

### Chatterbox TTS (Resemble AI)
- [GitHub Repository](https://github.com/resemble-ai/chatterbox)
- [Fine-tuning Toolkit](https://github.com/gokhaneraslan/chatterbox-finetuning)
- [Streaming & Fine-tuning](https://github.com/davidbrowne17/chatterbox-streaming)
- [Introducing Chatterbox Multilingual](https://www.resemble.ai/introducing-chatterbox-multilingual-open-source-tts-for-23-languages/)

### Similar Models
- [XTTS-v2 Documentation](https://docs.coqui.ai/en/latest/models/xtts.html)
- [XTTS Training Recipe](https://github.com/coqui-ai/TTS/blob/dev/recipes/ljspeech/xtts_v2/train_gpt_xtts.py)
- [AllTalk Simple Guide](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version))
- [AllTalk Advanced Guide](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Advanced-Version))
- [Kaggle XTTS Tutorial](https://www.kaggle.com/code/maxbr0wn/tutorial-fine-tuning-xttsv2-english)
- [F5-TTS Fine-tuning Discussion](https://github.com/SWivid/F5-TTS/discussions/57)
- [SpeechT5 Colab](https://colab.research.google.com/drive/1i7I5pzBcU3WDFarDnzweIj4-sVVoIUFJ)
