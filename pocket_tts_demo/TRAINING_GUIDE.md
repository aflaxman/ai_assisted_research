# Training & Voice Customization Guide for Pocket TTS

## Important: Training vs Voice Cloning

**Training/fine-tuning code is NOT publicly available** for Pocket TTS. The Kyutai team has released:
- ✅ Pre-trained 100M parameter model
- ✅ Inference code (generation and serving)
- ✅ Voice cloning capabilities (requires Hugging Face authentication)
- ✅ 8 preset voices (no authentication needed)
- ❌ Training scripts
- ❌ Dataset preparation code
- ❌ Fine-tuning procedures

## Two Ways to Customize Voices

### Method 1: Preset Voices (No Authentication)
Use the 8 built-in voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma
- ✅ Works out of the box
- ✅ No authentication needed
- ⚠️ Limited to preset options

### Method 2: Custom Voice Cloning (Requires Authentication)
Clone voices from your own audio files
- ✅ Unlimited customization
- ✅ Use any voice you want
- ⚠️ Requires Hugging Face account and authentication

**To enable voice cloning from files:**
```bash
# 1. Create a Hugging Face account at https://huggingface.co
# 2. Go to https://huggingface.co/kyutai/pocket-tts and accept the terms
# 3. Login locally:
uvx hf auth login
# Enter your HF token when prompted
```

Once authenticated, you can use custom WAV files for voice cloning!

## What IS Possible: Advanced Voice Cloning

Voice cloning is the primary way to customize Pocket TTS. It works by:
1. Providing a 5-30 second audio sample with desired characteristics
2. The model extracts and replicates: tone, accent, emotion, breathiness, grittiness
3. Generates new speech in that voice style

### Key Advantage
Voice cloning captures **acoustic properties** including:
- Vocal timbre and tone
- Breathiness and air flow
- Grittiness and vocal fry
- Room acoustics
- Microphone characteristics

## Creating Gritty/Breathy Voice Samples

### Option 1: Record Your Own Audio

**Equipment needed:**
- Microphone (phone mic works, but better mic = better results)
- Quiet environment
- Audio recording software

**Recording tips for gritty voice:**
```bash
# Record 10-30 seconds of speech with these characteristics:
# 1. Speak with vocal fry (that creaky, low register sound)
# 2. Lower your pitch slightly
# 3. Add slight roughness/raspiness to your voice
# 4. Avoid being too breathy (gritty ≠ breathy)
# 5. Speak clearly but with character
```

**Sample text to record:**
```
"The old road winds through forgotten places,
where shadows linger and secrets wait.
Every step echoes with stories untold,
carried on whispers of wind and dust."
```

Save as: `gritty_voice_sample.wav`

### Option 2: Process Existing Audio

If you have a recording but it's not gritty enough, you can use audio processing:

**Using Python (pydub and audio effects):**

```python
from pydub import AudioSegment
from pydub.effects import low_pass_filter, compress_dynamic_range
import numpy as np

# Load audio
audio = AudioSegment.from_wav("your_voice.wav")

# Make it grittier:
# 1. Slightly lower the pitch
audio_gritty = audio._spawn(audio.raw_data, overrides={
    "frame_rate": int(audio.frame_rate * 0.95)  # Lower by 5%
})

# 2. Add slight distortion via compression
audio_gritty = compress_dynamic_range(audio_gritty, threshold=-20.0, ratio=4.0)

# 3. Cut some high frequencies (makes it rougher)
audio_gritty = low_pass_filter(audio_gritty, 4000)

# Export
audio_gritty.export("gritty_voice_sample.wav", format="wav")
```

### Option 3: Find/Use Existing Gritty Audio

Sources for gritty voice samples:
- Public domain audiobooks (older recordings often have character)
- Creative Commons voice recordings
- Your own voice recordings with different styles
- Film dialogue clips (public domain films)

**Important:** Respect copyright! Only use audio you have rights to use.

## Using Your Custom Voice Sample

### Method 1: Python Script

```python
from pocket_tts import TTSModel
import scipy.io.wavfile

# Load model
tts_model = TTSModel.load_model()

# Use your custom gritty voice
voice_state = tts_model.get_state_for_audio_prompt("gritty_voice_sample.wav")

# Generate speech
text = "The wind howled through the canyon, a low whisper of secrets long forgotten."
audio = tts_model.generate_audio(voice_state, text)

# Save
scipy.io.wavfile.write("output_gritty.wav", tts_model.sample_rate, audio.numpy())
```

### Method 2: CLI

```bash
# Use your custom voice file directly
uvx pocket-tts generate \
    --voice gritty_voice_sample.wav \
    --text "Your text here" \
    --output-path output_gritty.wav
```

### Method 3: Use URLs

If your sample is hosted online:
```bash
uvx pocket-tts generate \
    --voice "https://example.com/my_gritty_voice.wav" \
    --text "Your text here"
```

## Advanced: Combining Multiple Techniques

For MAXIMUM grittiness, combine approaches:

1. **Start with a gritty preset voice** (like Javert)
2. **Extract and modify it:**

```python
# Generate sample with Javert
voice_state = tts_model.get_state_for_audio_prompt("javert")
audio = tts_model.generate_audio(voice_state, "Test recording for voice modification.")
scipy.io.wavfile.write("javert_sample.wav", tts_model.sample_rate, audio.numpy())

# Now process it to make it grittier (using pydub as shown above)
# Then use the processed version as your voice sample!
```

## Post-Processing Generated Audio

Even after generation, you can make output grittier:

**Using FFmpeg (command line):**
```bash
# Add slight distortion and lower frequencies
ffmpeg -i output.wav -af "acompressor=threshold=0.5:ratio=2:attack=20:release=250,equalizer=f=200:t=h:width=100:g=3" output_gritty.wav
```

**Using Python (pydub):**
```python
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range

audio = AudioSegment.from_wav("output.wav")

# Compress to add grit
audio = compress_dynamic_range(audio, threshold=-25.0, ratio=4.0, attack=5.0, release=50.0)

# Boost low frequencies
audio = audio.low_pass_filter(5000).apply_gain(2)

audio.export("output_gritty.wav", format="wav")
```

## Why Can't We Train the Model?

The Pocket TTS repository focuses on inference because:

1. **Training is expensive** - The larger Kyutai TTS model was trained with:
   - 32 H100 GPUs
   - 750k training steps
   - 2.5 million hours of audio data
   - Synthetic transcripts from Whisper

2. **Specialized infrastructure** - Training requires:
   - Large-scale audio datasets with transcripts
   - Distributed training setup
   - Weeks of compute time
   - Complex data preprocessing pipelines

3. **Research vs Production** - Kyutai released Pocket TTS for practical use, not research reproduction

## If You Really Need Training...

### Research Alternative: Delayed Streams Modeling
Check the related repository: [kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling)

This might have more research-oriented code, though training scripts aren't documented.

### Other Open-Source TTS with Training Code

If you absolutely need to train a TTS model:

1. **Coqui TTS** - Full training pipeline available
   ```bash
   pip install TTS
   # Training code and recipes available
   ```

2. **XTTS** - Fine-tuning supported
   - Can fine-tune on custom datasets
   - Lower resource requirements

3. **Piper TTS** - Training documentation available
   - Lightweight like Pocket TTS
   - Full training pipeline

### Fine-tuning Services

Some companies offer fine-tuning for TTS models:
- ElevenLabs (commercial)
- Play.ht (commercial)
- Resemble.ai (commercial)

## Best Practices for Voice Cloning

To get the best gritty/breathy results:

1. **Audio Quality**
   - Use WAV format (not MP3)
   - 24kHz or higher sample rate
   - Mono or stereo both work
   - Minimal background noise

2. **Voice Characteristics**
   - 10-30 seconds is optimal
   - Speak naturally with desired characteristics
   - Include varied intonation
   - Consistency matters (don't switch styles mid-sample)

3. **Content**
   - Use complete sentences
   - Natural speech rhythm
   - Avoid dead air or long pauses
   - Phonetically diverse text helps

4. **Iteration**
   - Try multiple samples
   - Experiment with different processing
   - Combine techniques
   - Test different source voices

## Example Workflow

Here's a complete workflow to create a custom gritty voice:

```bash
cd /home/user/ai_assisted_research/pocket_tts_demo

# 1. Start with a preset voice
python create_custom_voice.py

# 2. This will:
#    - Generate audio with Javert (grittiest preset)
#    - Process it to be even grittier
#    - Use it as a custom voice
#    - Generate final output

# 3. Listen and iterate
explorer.exe output
```

See `create_custom_voice.py` for the complete implementation!

## Summary

| Method | Difficulty | Quality | Control |
|--------|-----------|---------|---------|
| Use preset voices | Easy | Good | Low |
| Voice cloning with your recording | Medium | Very Good | High |
| Audio processing + cloning | Hard | Excellent | Very High |
| Training from scratch | Not Available | N/A | N/A |

**Recommendation:** Start with voice cloning using a custom audio sample. This gives you 90% of what training would provide, without the complexity!

## Sources

- [Pocket TTS GitHub](https://github.com/kyutai-labs/pocket-tts)
- [Kyutai Blog: Pocket TTS](https://kyutai.org/blog/2026-01-13-pocket-tts)
- [Kyutai Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- [Pocket TTS on Hugging Face](https://huggingface.co/kyutai/pocket-tts)
