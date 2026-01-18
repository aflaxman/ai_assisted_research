#!/usr/bin/env python3
"""
Explore the voice_state object to understand what parameters are extracted.
"""

from pocket_tts import TTSModel
import torch
import numpy as np

print("Loading model...")
tts = TTSModel.load_model()

# Get voice state from a preset voice
print("\nExtracting voice state from 'javert' preset...")
voice_state = tts.get_state_for_audio_prompt("javert")

print(f"\nType of voice_state: {type(voice_state)}")

# Check if it's a dict, tensor, or other object
if isinstance(voice_state, dict):
    print("\nvoice_state is a dictionary with keys:")
    for key, value in voice_state.items():
        if torch.is_tensor(value):
            print(f"  {key}: Tensor of shape {value.shape}, dtype {value.dtype}")
            print(f"         Range: [{value.min().item():.4f}, {value.max().item():.4f}]")
            print(f"         Mean: {value.mean().item():.4f}, Std: {value.std().item():.4f}")
        else:
            print(f"  {key}: {type(value)} = {value}")

elif torch.is_tensor(voice_state):
    print(f"\nvoice_state is a Tensor:")
    print(f"  Shape: {voice_state.shape}")
    print(f"  Dtype: {voice_state.dtype}")
    print(f"  Range: [{voice_state.min().item():.4f}, {voice_state.max().item():.4f}]")
    print(f"  Mean: {voice_state.mean().item():.4f}, Std: {voice_state.std().item():.4f}")

    # Check if it's a sequence or embedding
    if len(voice_state.shape) == 2:
        print(f"\n  This looks like a sequence of {voice_state.shape[0]} frames")
        print(f"  with {voice_state.shape[1]} dimensions each")

    elif len(voice_state.shape) == 3:
        print(f"\n  This looks like a batch of {voice_state.shape[0]} sequences")
        print(f"  with {voice_state.shape[1]} frames and {voice_state.shape[2]} dims")

else:
    print(f"\nvoice_state is a {type(voice_state)} object")
    print(f"Attributes: {dir(voice_state)}")

# Check the actual class and methods
print(f"\nvoice_state class: {voice_state.__class__.__name__}")
print(f"Module: {voice_state.__class__.__module__}")

# Try to understand the structure
if hasattr(voice_state, '__dict__'):
    print("\nObject attributes:")
    for attr, value in voice_state.__dict__.items():
        if torch.is_tensor(value):
            print(f"  {attr}: Tensor {value.shape}")
        else:
            print(f"  {attr}: {type(value)}")

# Check if there are any parameters we can access
print("\n" + "="*60)
print("Checking for accessible parameters...")

# Common attributes that might exist
attrs_to_check = ['hidden_states', 'embeddings', 'latents', 'features',
                  'kv_cache', 'cache', 'state', 'past_key_values']

for attr in attrs_to_check:
    if hasattr(voice_state, attr):
        val = getattr(voice_state, attr)
        print(f"✓ Found: {attr} = {type(val)}")
        if torch.is_tensor(val):
            print(f"  Shape: {val.shape}")

print("="*60)
