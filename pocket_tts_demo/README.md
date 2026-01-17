# Pocket TTS Demo - Breathy & Gritty Voice Generator

A simple demo application for [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) that generates speech with different voice styles, focusing on breathy and gritty tones.

## 🎯 What This Demo Does

1. Generates speech using multiple preset voices
2. Saves audio files for comparison
3. Helps you find voices with breathy, gritty characteristics
4. **NEW:** Advanced voice customization with audio processing
5. **NEW:** Use your own voice recordings for custom TTS

## 📄 What's Included

- **`tts_demo.py`** - Basic demo with preset voices
- **`create_custom_voice.py`** - Advanced: Create grittier custom voices
- **`record_voice.py`** - Record your own voice (Linux/Mac)
- **`convert_recording.py`** - Convert audio files to TTS format (for WSL users)
- **`use_your_voice.py`** - Use your own audio recordings
- **`output/`** - Pre-generated example files

## 📋 Prerequisites

- Python 3.10 or later
- `uv` package manager (recommended) or `pip`

## ⚡ Quick Start Guide

### Which Script Should I Use?

```
Script                    Purpose                      Auth?  Best For
─────────────────────────────────────────────────────────────────────────────
tts_demo.py               Test preset voices           No     Getting started
create_custom_voice.py    Compare & enhance voices     No     Finding grittiest presets
record_voice.py           Record voice (Linux/Mac)     No     Direct recording
convert_recording.py      Convert audio to TTS format  No     WSL users, format conversion
use_your_voice.py         Clone custom voice           Yes*   Using recorded samples
```

*Voice cloning from custom files requires Hugging Face authentication (see Method 2 below)

### Fastest Way to Get Started

```bash
cd pocket_tts_demo

# Install ffmpeg (required for audio playback and processing)
sudo apt update && sudo apt install -y ffmpeg

# One-time setup: create venv and install
uv venv && source .venv/bin/activate && uv pip install pocket-tts scipy pydub

# Run the basic demo
python tts_demo.py

# Listen to the results
ffplay output/javert_demo.wav
```

**Already have audio files?** The `output/` folder contains pre-generated examples - just play them with `ffplay output/javert_demo.wav`!

## 🚀 Full Step-by-Step Instructions

### Step 1: Install uv (if you don't have it)

```bash
# On Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Install ffmpeg (Required)

```bash
# Install ffmpeg for audio playback and processing
sudo apt update
sudo apt install -y ffmpeg
```

### Step 3: Create Virtual Environment and Install Dependencies

```bash
cd pocket_tts_demo

# Create isolated virtual environment
uv venv

# Activate it
source .venv/bin/activate  # On Linux/WSL/Mac
# OR on Windows: .venv\Scripts\activate

# Install dependencies in the isolated environment
uv pip install pocket-tts scipy pydub
```

**Alternative:** If you don't have `uv`, you can use standard Python:
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### Step 4: Run the Demo

```bash
python tts_demo.py
```

**What to expect:**
- First run will download the model (~200MB) - this takes a minute
- The script will generate 4 audio files with different voices
- Files will be saved in the `output/` directory

### Step 5: Listen to the Results

**Play the generated audio files:**
```bash
# Play the grittiest voice
ffplay output/javert_demo.wav

# Play the breathy voice
ffplay output/marius_demo.wav

# Play all voices for comparison (press 'q' to skip to next file)
ffplay output/alba_demo.wav output/marius_demo.wav output/javert_demo.wav output/jean_demo.wav
```

**What to listen for:**
- **Javert** - Deeper and grittier tone
- **Marius** - Breathy qualities
- **Alba & Jean** - Contrast voices for comparison

## 🎨 Customizing for More Gritty/Breathy Sound

### Option 1: Try Different Voices

Edit `tts_demo.py` and add more voices to the list:

```python
voices = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
```

### Option 2: Use Your Own Voice Sample (Advanced)

If you have a WAV file with the breathy/gritty voice you want:

1. Place your WAV file in the demo directory (e.g., `my_voice.wav`)
2. Modify the script:

```python
# Replace this line:
voice_prompt = f"hf://kyutai/tts-voices/{voice}/casual.wav"

# With this:
voice_prompt = "my_voice.wav"
```

The model will clone the voice style from your audio!

### Option 3: Quick Test with CLI

Want to test quickly without the script?

```bash
# Try different voices
uvx pocket-tts generate --voice javert --text "The wind howled through the canyon."
uvx pocket-tts generate --voice marius --text "The wind howled through the canyon."

# Use your own voice file
uvx pocket-tts generate --voice my_voice.wav --text "The wind howled through the canyon."
```

## 🔥 Advanced: Create Even Grittier Voices

### Method 1: Automated Voice Processing

Use the included script to automatically create a grittier voice:

```bash
# Make sure you're in your virtual environment first!
# Then run:
python create_custom_voice.py
```

This script will:
1. Start with the Javert preset (grittiest default)
2. Apply subtle audio processing (compression, filtering)
3. Generate comparison samples across different voices

**Note:** Audio processing is optional and minimal to preserve quality. For significant voice changes, use Method 2 (record your own voice).

### Method 2: Record and Clone Your Own Voice

**This is the most powerful way to create a custom gritty/breathy voice!**

#### Step 1: Authenticate with Hugging Face (Required for custom voice cloning)

```bash
# 1. Create a free account at https://huggingface.co
# 2. Go to https://huggingface.co/kyutai/pocket-tts and accept the terms
# 3. Login locally:
uvx hf auth login
# Enter your HF token when prompted
```

**Note:** The 8 preset voices (alba, marius, javert, etc.) work WITHOUT authentication. You only need HF auth for cloning custom voices from WAV files.

#### Step 2: Record Your Voice

**⚠️ WSL Users:** WSL doesn't have direct access to Windows microphones. Use Option A or B below.

**Option A: Record on Windows (Recommended for WSL)**

1. **Record on Windows:**
   - Open Windows Voice Recorder or download Audacity
   - Record 15-30 seconds of speech in your desired voice style
   - Speak the suggested text (see tips below)

2. **Copy to WSL:**
   ```bash
   # Copy from Windows Downloads to WSL
   cp /mnt/c/Users/YourUsername/Downloads/recording.wav .
   ```

3. **Convert to correct format:**
   ```bash
   python convert_recording.py recording.wav my_voice.wav
   ```

**Option B: Record directly in WSL (Linux/Mac)**

Install recording dependency:
```bash
uv pip install sounddevice
sudo apt-get install portaudio19-dev
```

Record:
```bash
python record_voice.py
```

**Tips for recording:**
- Use a quiet room with minimal background noise
- Speak naturally with varied intonation (don't be monotone)
- **For gritty voice:** Lower your pitch, add vocal fry (that creaky sound), slight raspiness
- **For breathy voice:** Emphasize air flow, speak softer, use more breath
- Record 10-30 seconds - enough for variety but not too long
- Use complete sentences with natural rhythm

**Suggested text to read:**
```
"The old road winds through forgotten places,
where shadows linger and secrets wait.
Every step echoes with stories untold,
carried on whispers of wind and dust."
```

#### Step 3: Use Your Recording for Voice Cloning

```bash
# Test your recording first
ffplay my_voice.wav

# Use it to generate speech
python use_your_voice.py my_voice.wav

# Or use directly with CLI:
uvx pocket-tts generate --voice my_voice.wav --text "Your text here"
```

The model will extract and replicate:
- Vocal timbre and tone
- Breathiness and air flow
- Grittiness and vocal fry
- Pitch characteristics
- Accent and speaking style

#### Recording Tips for Maximum Grittiness

1. **Vocal Fry**: That low, creaky sound at the bottom of your vocal range
2. **Lower Pitch**: Speak 5-10% lower than your normal speaking voice
3. **Consonant Emphasis**: Hit consonants like "t", "k", "s" harder
4. **Slight Raspiness**: Add a bit of roughness, but don't strain your voice
5. **Consistent Style**: Maintain the same character throughout the recording

### Method 3: Can I Train My Own Model?

**No - training code is NOT publicly available.** The Kyutai team released:
- ✅ Pre-trained 100M parameter model
- ✅ Inference code
- ✅ Voice cloning from audio samples
- ❌ Training scripts or datasets

**Why training isn't available:**
- Training the larger model required 32 H100 GPUs, 750k training steps, and 2.5 million hours of audio
- Specialized infrastructure and expensive compute
- Research code vs. production release

**Good news:** Voice cloning provides 90% of what training would give you! You can clone ANY voice from a 10-30 second sample.

## 🎛️ Tips for Best Results

1. **Voice Selection**: Deeper male voices (like Javert) tend to sound grittier
2. **Text Choice**: Use text with consonants like "h", "s", "th" to emphasize breathy qualities
3. **Voice Cloning**: For truly custom breathy/gritty voice, record a 10-30 second WAV sample with the characteristics you want
4. **Audio Quality**: Use WAV format, 24kHz+ sample rate, minimal background noise
5. **Voice Characteristics**: Speak with vocal fry or slight raspiness for gritty effect

## 🐛 Troubleshooting

**Model download is slow**: First run downloads ~200MB. Be patient!

**No audio output**: Make sure dependencies are installed: `uv pip install pocket-tts scipy pydub`

**Import errors**: Create a fresh virtual environment and re-run Step 3

**Audio won't play**: Make sure ffmpeg is installed: `sudo apt install ffmpeg`

**Recording issues on WSL**: See README Method 2 → Step 2 for WSL-specific recording instructions

## 📚 Learn More

- [Pocket TTS Repository](https://github.com/kyutai-labs/pocket-tts)
- [Documentation](https://github.com/kyutai-labs/pocket-tts/blob/main/README.md)

## 🎉 Next Steps

Once you find a voice you like:
- Experiment with different text
- Try voice cloning with your own samples
- Integrate into your projects using the Python API

Enjoy creating unique vocal experiences! 🎤
