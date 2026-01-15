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
- **`use_your_voice.py`** - Use your own audio recordings
- **`TRAINING_GUIDE.md`** - Complete guide to voice customization
- **`output/`** - Pre-generated example files

## 📋 Prerequisites

- Python 3.10 or later
- `uv` package manager (recommended) or `pip`

## ⚡ Quick Start Guide

### Which Script Should I Use?

| Script | Purpose | Auth Needed? | Best For |
|--------|---------|--------------|----------|
| `tts_demo.py` | Test preset voices | No | Getting started |
| `create_custom_voice.py` | Compare & enhance voices | No | Finding grittiest presets |
| `use_your_voice.py` | Clone YOUR voice | Yes* | Custom voice cloning |

*Requires Hugging Face authentication - see TRAINING_GUIDE.md

### Fastest Way to Get Started

Already have the dependencies installed? Just run:
```bash
cd pocket_tts_demo
python tts_demo.py  # Generate samples with 4 voices
# OR
python create_custom_voice.py  # Advanced comparison with all voices
```

Then open the `output/` folder and listen to the WAV files!

## 🚀 Full Step-by-Step Instructions

### Step 1: Install uv (if you don't have it)

```bash
# On Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Install Dependencies

```bash
cd pocket_tts_demo
uv pip install pocket-tts scipy
```

Or with regular pip:
```bash
pip install pocket-tts scipy
```

### Step 3: Run the Demo

```bash
python tts_demo.py
```

**What to expect:**
- First run will download the model (~200MB) - this takes a minute
- The script will generate 4 audio files with different voices
- Files will be saved in the `output/` directory

### Step 4: Listen to the Results

Open the `output/` folder and play the WAV files:

```bash
# On WSL, you can open the folder in Windows Explorer
explorer.exe output

# Or play directly (if you have a player installed)
mpv output/javert_demo.wav
# or
aplay output/javert_demo.wav
```

Listen for:
- **Javert** - Usually deeper and grittier
- **Marius** - Often has breathy qualities
- **Alba & Jean** - Compare these for contrast

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
python create_custom_voice.py
```

This script will:
1. Start with the Javert preset (grittiest default)
2. Process it with audio effects (pitch, compression, filtering)
3. Create an even grittier custom voice
4. Generate comparison samples

**Requires:** `pip install pydub` and `sudo apt-get install ffmpeg`

### Method 2: Use Your Own Voice Recording

Record your own gritty/breathy voice and use it:

```bash
# Record 10-30 seconds of speech with desired characteristics
# Save as: my_voice.wav

# Use it to generate speech
python use_your_voice.py my_voice.wav
```

The model will clone your voice's characteristics!

### Method 3: Train Your Own Model?

**Important:** Training code is NOT publicly available for Pocket TTS. However, voice cloning provides 90% of what custom training would give you!

See **`TRAINING_GUIDE.md`** for:
- Why training isn't available
- Detailed voice cloning techniques
- Audio processing methods
- Alternative TTS models with training support
- Tips for recording the perfect gritty voice sample

## 🎛️ Tips for Best Results

1. **Voice Selection**: Deeper male voices (like Javert) tend to sound grittier
2. **Text Choice**: Use text with consonants like "h", "s", "th" to emphasize breathy qualities
3. **Voice Cloning**: For truly custom breathy/gritty voice, record a 10-30 second WAV sample with the characteristics you want
4. **Audio Quality**: Use WAV format, 24kHz+ sample rate, minimal background noise
5. **Voice Characteristics**: Speak with vocal fry or slight raspiness for gritty effect

## 🐛 Troubleshooting

**Model download is slow**: First run downloads ~200MB. Be patient!

**No audio output**: Make sure scipy is installed: `pip install scipy`

**Import errors**: Try using `uv` instead of pip, or create a fresh virtual environment

**WSL audio issues**: Audio files should play fine in Windows (use `explorer.exe output`)

## 📚 Learn More

- [Pocket TTS Repository](https://github.com/kyutai-labs/pocket-tts)
- [Documentation](https://github.com/kyutai-labs/pocket-tts/blob/main/README.md)

## 🎉 Next Steps

Once you find a voice you like:
- Experiment with different text
- Try voice cloning with your own samples
- Integrate into your projects using the Python API

Enjoy creating unique vocal experiences! 🎤
