#!/usr/bin/env python3
"""Quick test of advanced voice mixer functionality."""

import sys
import numpy as np

# Test imports
print("Testing imports...")
try:
    from pocket_tts import TTSModel
    import gradio as gr
    import torch
    import librosa
    import plotly.graph_objects as go
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test model loading
print("\nTesting model loading...")
try:
    tts = TTSModel.load_model()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading error: {e}")
    sys.exit(1)

# Test voice state extraction
print("\nTesting voice state extraction...")
try:
    voice_state = tts.get_state_for_audio_prompt("javert")
    print(f"✓ Voice state extracted: {type(voice_state)}")
except Exception as e:
    print(f"✗ Voice state error: {e}")
    sys.exit(1)

# Test audio generation
print("\nTesting audio generation...")
try:
    audio = tts.generate_audio(voice_state, "Test.")
    if torch.is_tensor(audio):
        audio = audio.cpu().numpy()
    print(f"✓ Generated {len(audio)} audio samples at {tts.sample_rate}Hz")
    duration = len(audio) / tts.sample_rate
    print(f"  Duration: {duration:.2f}s")
except Exception as e:
    print(f"✗ Generation error: {e}")
    sys.exit(1)

# Test pitch/tempo processing
print("\nTesting pitch/tempo processing...")
try:
    # Test pitch shift
    pitched = librosa.effects.pitch_shift(audio.astype(np.float32), sr=tts.sample_rate, n_steps=2)
    print(f"✓ Pitch shift: {len(pitched)} samples")

    # Test tempo change
    tempo = librosa.effects.time_stretch(audio.astype(np.float32), rate=1.2)
    print(f"✓ Tempo change: {len(tempo)} samples")
except Exception as e:
    print(f"✗ Pitch/tempo error: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("✓ All tests passed!")
print("="*50)
