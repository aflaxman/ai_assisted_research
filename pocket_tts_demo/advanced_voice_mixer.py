#!/usr/bin/env python3
"""
Advanced Voice Mixer - Interactive tool to blend and manipulate voice characteristics.

This tool extracts voice embeddings from audio files and allows you to:
- Blend multiple voices together
- Interpolate between different voice characteristics
- Modify KV cache values to tweak voice properties
- Visualize voice embeddings

## How Voice State Parameters Map to Code

When you provide custom audio, the model extracts a "voice_state" which is a dictionary
containing the transformer's KV (key-value) cache:

```python
voice_state = {
    'transformer.layers.0.self_attn': {
        'current_end': Tensor[125],  # Position tracker
        'cache': Tensor[125, 2, 8, 127, 256]  # Actual KV cache
    },
    'transformer.layers.1.self_attn': { ... },
    ...  # 6 layers total for Pocket TTS
}
```

The cache dimensions are:
- Dim 0 (125): Batch size
- Dim 1 (2): Keys and Values
- Dim 2 (8): Number of attention heads
- Dim 3 (127): Sequence length (audio frames)
- Dim 4 (256): Embedding dimension per head

## What the Exposed Parameters Do

### Voice Blending (blend_ratio)
**Code:**
```python
blended_cache = (1 - blend_ratio) * cache_A + blend_ratio * cache_B
```
**What it does:** Weighted average of attention patterns from two voices.
- 0.0 = 100% Voice A
- 0.5 = 50/50 mix
- 1.0 = 100% Voice B

**Effect:** Combines voice characteristics like pitch, timbre, rhythm, breathiness.

### Temperature Modification
**Code:**
```python
mean = torch.nanmean(cache)
modified = mean + (cache - mean) * temperature
```
**What it does:** Scales variance around the mean.
- temperature > 1.0: Amplifies deviations → more varied/dynamic voice
- temperature < 1.0: Reduces deviations → more consistent/flat voice
- temperature = 1.0: No change

**Effect:** Controls how much the voice varies during speech. Higher temperature
makes the voice more expressive and dynamic, lower makes it more monotone.

**Maps to:** The variance of attention weights in each layer's cache.

### Sharpness Modification
**Code:**
```python
median = torch.nanmedian(cache)
sign = torch.sign(cache - median)
modified = cache + sharpness * sign * torch.abs(cache - median)
```
**What it does:** Enhances or reduces contrast relative to the median.
- sharpness > 0: Amplifies differences from median → sharper/crisper voice
- sharpness < 0: Reduces differences → softer/mellower voice
- sharpness = 0: No change

**Effect:** Controls voice clarity and distinctiveness. Positive values make
consonants crisper and vowels more defined. Negative values create a softer,
more muffled quality.

**Maps to:** The distribution of attention weights - sharper = more peaked
attention, softer = more diffuse attention.

### Depth Modification
**Code:**
```python
modified = cache + depth
```
**What it does:** Shifts all cache values by a constant.
- depth > 0: Shift upward → deeper/darker resonance
- depth < 0: Shift downward → lighter/brighter resonance
- depth = 0: No change

**Effect:** Controls the overall tonal quality. Positive depth creates a
deeper, more resonant voice (like shifting pitch down). Negative depth
creates a lighter, airier voice.

**Maps to:** The mean baseline of all attention values in the cache.

## Limitations and Caveats

1. **Not trained for this:** The model wasn't trained to handle modified KV caches,
   so extreme values may produce artifacts or degraded quality.

2. **Heuristic transformations:** These modifications are educated guesses about
   how KV cache statistics relate to voice perception. They work reasonably well
   but aren't guaranteed.

3. **Non-linear interactions:** Combining multiple modifications may have
   unexpected effects due to non-linear interactions in the transformer.

4. **Better alternatives exist:** For production use, fine-tuning the model on
   target speech data (see FINE_TUNING_ANALYSIS.md) will give better results.

5. **Voice cloning limits:** Custom audio files require HF authentication for
   voice cloning weights. Without it, only preset voices work.

This tool is for experimentation and exploration of the voice embedding space!
"""

import gradio as gr
import torch
import numpy as np
from pocket_tts import TTSModel
import scipy.io.wavfile
from pathlib import Path
import json

# Initialize model
print("Loading Pocket TTS model...")
tts = TTSModel.load_model()
print("Model loaded successfully!")

# ACTUAL preset voices (verified to exist)
PRESET_VOICES = [
    "alba",      # Female
    "marius",    # Male, breathy
    "javert",    # Male, deep/gritty
    "jean",      # Male
    "fantine",   # Female
    "cosette",   # Female
    "eponine",   # Female
    "azelma",    # Female
]

# Voice descriptions
VOICE_INFO = {
    "alba": "Female, clear",
    "marius": "Male, breathy",
    "javert": "Male, deep/gritty",
    "jean": "Male, neutral",
    "fantine": "Female, mature",
    "cosette": "Female, young",
    "eponine": "Female, spirited",
    "azelma": "Female, soft",
}

# Default text samples
SAMPLE_TEXTS = {
    "Short": "The quick brown fox jumps over the lazy dog.",
    "Medium": "In the beginning was the Word, and the Word was with God, and the Word was made flesh.",
    "Long": "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
    "Atmospheric": "The ancient forest whispers secrets through rustling leaves, while shadows dance in the fading light.",
}

# Cache for voice states
voice_cache = {}


def get_voice_state(voice_source):
    """Get or cache voice state."""
    if voice_source not in voice_cache:
        print(f"Extracting voice state for: {voice_source}")
        voice_cache[voice_source] = tts.get_state_for_audio_prompt(voice_source)
    return voice_cache[voice_source]


def analyze_voice_state(voice_state):
    """Analyze the structure and statistics of a voice state."""
    info = []
    info.append("# Voice State Analysis\n")

    total_params = 0
    for key, value in voice_state.items():
        if isinstance(value, dict) and 'cache' in value:
            cache = value['cache']
            if torch.is_tensor(cache):
                info.append(f"## {key}")
                info.append(f"- Shape: {list(cache.shape)}")
                info.append(f"- Total values: {cache.numel():,}")
                info.append(f"- Range: [{cache.min().item():.4f}, {cache.max().item():.4f}]")
                info.append(f"- Mean: {cache.mean().item():.4f}")
                info.append(f"- Std: {cache.std().item():.4f}")
                info.append("")
                total_params += cache.numel()

    info.insert(1, f"**Total cache size**: {total_params:,} values\n")
    return "\n".join(info)


def extract_voice_vector(voice_state):
    """Extract a summary vector from voice state by averaging key statistics."""
    vectors = []
    for key, value in sorted(voice_state.items()):
        if isinstance(value, dict) and 'cache' in value:
            cache = value['cache']
            if torch.is_tensor(cache):
                # Extract statistical summary
                # Only use non-NaN values
                mask = ~torch.isnan(cache)
                valid_cache = cache[mask]

                if valid_cache.numel() > 0:
                    stats = torch.tensor([
                        valid_cache.mean().item(),
                        valid_cache.std().item(),
                        valid_cache.min().item(),
                        valid_cache.max().item(),
                        valid_cache.median().item(),
                    ])
                    vectors.append(stats)

    if vectors:
        return torch.cat(vectors)
    return None


def blend_voice_states(voice_state_a, voice_state_b, blend_ratio):
    """
    Blend two voice states together.

    Args:
        voice_state_a: First voice state (dict)
        voice_state_b: Second voice state (dict)
        blend_ratio: How much of B to blend in (0.0 = all A, 1.0 = all B)

    Returns:
        Blended voice state
    """
    blended = {}

    for key in voice_state_a.keys():
        if key in voice_state_b:
            value_a = voice_state_a[key]
            value_b = voice_state_b[key]

            if isinstance(value_a, dict) and isinstance(value_b, dict):
                blended[key] = {}
                for subkey in value_a.keys():
                    if subkey in value_b:
                        if torch.is_tensor(value_a[subkey]) and torch.is_tensor(value_b[subkey]):
                            # Blend tensors
                            tensor_a = value_a[subkey]
                            tensor_b = value_b[subkey]

                            # Handle NaN values
                            mask_a = ~torch.isnan(tensor_a)
                            mask_b = ~torch.isnan(tensor_b)

                            blended_tensor = torch.where(
                                mask_a & mask_b,
                                (1 - blend_ratio) * tensor_a + blend_ratio * tensor_b,
                                torch.where(mask_a, tensor_a, tensor_b)
                            )
                            blended[key][subkey] = blended_tensor
                        else:
                            # For non-tensor values, just take from A
                            blended[key][subkey] = value_a[subkey]
            else:
                blended[key] = value_a
        else:
            blended[key] = voice_state_a[key]

    return blended


def modify_voice_state(voice_state, temperature, sharpness, depth):
    """
    Modify voice state characteristics.

    Args:
        voice_state: Input voice state
        temperature: Controls variability (higher = more varied)
        sharpness: Controls distinctiveness (higher = sharper)
        depth: Controls resonance (deeper values)

    Returns:
        Modified voice state
    """
    modified = {}

    for key, value in voice_state.items():
        if isinstance(value, dict):
            modified[key] = {}
            for subkey, subvalue in value.items():
                if torch.is_tensor(subvalue) and subkey == 'cache':
                    # Apply modifications
                    tensor = subvalue.clone()

                    # Temperature: scale variance
                    if temperature != 1.0:
                        mean = torch.nanmean(tensor)
                        tensor = mean + (tensor - mean) * temperature

                    # Sharpness: enhance contrast
                    if sharpness != 0.0:
                        median = torch.nanmedian(tensor)
                        tensor = tensor + sharpness * torch.sign(tensor - median) * torch.abs(tensor - median)

                    # Depth: shift mean
                    if depth != 0.0:
                        tensor = tensor + depth

                    modified[key][subkey] = tensor
                else:
                    modified[key][subkey] = subvalue
        else:
            modified[key] = value

    return modified


def generate_with_custom_voice(text, voice_a, voice_b, blend_ratio,
                                temperature, sharpness, depth,
                                use_preset_a, use_preset_b,
                                custom_audio_a, custom_audio_b):
    """Generate speech with blended/modified voice."""
    if not text or not text.strip():
        return None, "⚠️ Please enter some text"

    try:
        # Get voice states
        if use_preset_a and voice_a:
            state_a = get_voice_state(voice_a)
            source_a = f"{voice_a} ({VOICE_INFO.get(voice_a, '')})"
        elif custom_audio_a is not None:
            state_a = tts.get_state_for_audio_prompt(custom_audio_a)
            source_a = "Custom audio A"
        else:
            return None, "⚠️ Please select Voice A"

        # Check if we're blending
        if blend_ratio > 0:
            if use_preset_b and voice_b:
                state_b = get_voice_state(voice_b)
                source_b = f"{voice_b} ({VOICE_INFO.get(voice_b, '')})"
            elif custom_audio_b is not None:
                state_b = tts.get_state_for_audio_prompt(custom_audio_b)
                source_b = "Custom audio B"
            else:
                return None, "⚠️ Please select Voice B for blending"

            # Blend voices
            print(f"Blending {source_a} ({1-blend_ratio:.0%}) + {source_b} ({blend_ratio:.0%})")
            voice_state = blend_voice_states(state_a, state_b, blend_ratio)
            voice_info = f"🎨 Blended: {source_a} ({(1-blend_ratio)*100:.0f}%) + {source_b} ({blend_ratio*100:.0f}%)"
        else:
            voice_state = state_a
            voice_info = f"🎭 Using: {source_a}"

        # Apply modifications
        if temperature != 1.0 or sharpness != 0.0 or depth != 0.0:
            print(f"Modifying voice: temp={temperature}, sharp={sharpness}, depth={depth}")
            voice_state = modify_voice_state(voice_state, temperature, sharpness, depth)
            voice_info += f"\n🔧 Modified: temp={temperature:.2f}, sharp={sharpness:.2f}, depth={depth:.2f}"

        # Generate audio
        audio = tts.generate_audio(voice_state, text)

        # Convert to numpy
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()

        status = f"✅ Generated {len(audio)/tts.sample_rate:.2f}s of audio\n{voice_info}"

        return (tts.sample_rate, audio), status

    except Exception as e:
        import traceback
        return None, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"


def compare_voices(voice_a, voice_b):
    """Compare two voice states."""
    try:
        state_a = get_voice_state(voice_a)
        state_b = get_voice_state(voice_b)

        vec_a = extract_voice_vector(state_a)
        vec_b = extract_voice_vector(state_b)

        if vec_a is not None and vec_b is not None:
            # Compute similarity
            similarity = torch.nn.functional.cosine_similarity(
                vec_a.unsqueeze(0), vec_b.unsqueeze(0)
            ).item()

            distance = torch.norm(vec_a - vec_b).item()

            info = f"# Voice Comparison\n\n"
            info += f"**{voice_a}** vs **{voice_b}**\n\n"
            info += f"- Cosine Similarity: {similarity:.4f} ({similarity*100:.1f}% similar)\n"
            info += f"- Euclidean Distance: {distance:.4f}\n"
            info += f"- Vector A size: {vec_a.numel()} dimensions\n"
            info += f"- Vector B size: {vec_b.numel()} dimensions\n"

            return info
        else:
            return "Could not extract voice vectors for comparison"

    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface with auto-generation
with gr.Blocks(title="Advanced Voice Mixer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎛️ Advanced Voice Mixer Studio

    Blend voices, modify characteristics, and explore the voice embedding space in real-time.

    **Controls update automatically** - just move sliders and hear the changes!
    """)

    with gr.Tabs():
        # Main mixer tab
        with gr.Tab("🎨 Voice Mixer"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Text Input")
                    text_input = gr.Textbox(
                        label="Text to Speak",
                        placeholder="Enter text here...",
                        lines=4,
                        value="The ancient forest whispers secrets through rustling leaves.",
                    )

                    with gr.Row():
                        for name in SAMPLE_TEXTS.keys():
                            gr.Button(name, size="sm").click(
                                fn=lambda n=name: SAMPLE_TEXTS[n],
                                outputs=text_input,
                            )

                    gr.Markdown("### Voice Blending")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Voice A**")
                            use_preset_a = gr.Checkbox(label="Use preset voice", value=True)
                            voice_a = gr.Dropdown(
                                label="Preset A",
                                choices=PRESET_VOICES,
                                value="javert",
                                info=VOICE_INFO.get("javert", ""),
                            )
                            custom_audio_a = gr.Audio(
                                label="Or upload custom audio A",
                                type="filepath",
                            )

                        with gr.Column():
                            gr.Markdown("**Voice B**")
                            use_preset_b = gr.Checkbox(label="Use preset voice", value=True)
                            voice_b = gr.Dropdown(
                                label="Preset B",
                                choices=PRESET_VOICES,
                                value="marius",
                                info=VOICE_INFO.get("marius", ""),
                            )
                            custom_audio_b = gr.Audio(
                                label="Or upload custom audio B",
                                type="filepath",
                            )

                    blend_ratio = gr.Slider(
                        label="Voice Blend (0 = all A, 1 = all B)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        info="Slide to blend between Voice A and Voice B"
                    )

                    gr.Markdown("### Voice Modifications")

                    temperature = gr.Slider(
                        label="Temperature (voice variability)",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="Higher = more varied/dynamic, Lower = more consistent"
                    )

                    sharpness = gr.Slider(
                        label="Sharpness (voice distinctiveness)",
                        minimum=-1.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.1,
                        info="Higher = sharper/crisper, Lower = softer/mellower"
                    )

                    depth = gr.Slider(
                        label="Depth (voice resonance)",
                        minimum=-0.5,
                        maximum=0.5,
                        value=0.0,
                        step=0.05,
                        info="Higher = deeper/darker, Lower = lighter/brighter"
                    )

                    enable_loop = gr.Checkbox(
                        label="Loop playback",
                        value=True,
                        info="Automatically loop the generated audio"
                    )

                with gr.Column():
                    status_text = gr.Markdown("Adjust controls to generate speech automatically!")

                    audio_output = gr.Audio(
                        label="Generated Audio",
                        type="numpy",
                        autoplay=True,
                        loop=True,
                    )

                    gr.Markdown("""
                    ### 💡 Tips

                    **Blending:**
                    - Start with blend=0 to hear Voice A pure
                    - Gradually increase to hear the blend
                    - Try: javert (gritty) + marius (breathy)

                    **Modifications:**
                    - Temperature: 1.0 = original, >1.0 = more dynamic
                    - Sharpness: 0 = original, >0 = crisper
                    - Depth: 0 = original, >0 = deeper

                    **Loop playback:** Enabled by default for easier comparison
                    """)

            # Auto-generate when any parameter changes
            inputs = [
                text_input,
                voice_a, voice_b, blend_ratio,
                temperature, sharpness, depth,
                use_preset_a, use_preset_b,
                custom_audio_a, custom_audio_b,
            ]

            # Connect all inputs to trigger auto-generation
            for inp in inputs:
                inp.change(
                    fn=generate_with_custom_voice,
                    inputs=inputs,
                    outputs=[audio_output, status_text],
                )

        # Documentation tab
        with gr.Tab("📖 Parameter Documentation"):
            gr.Markdown("""
            ## How Voice State Parameters Map to Code

            ### Voice State Structure

            When you provide custom audio, the model extracts a "voice_state" dictionary
            containing the transformer's KV (key-value) cache:

            ```python
            voice_state = {
                'transformer.layers.0.self_attn': {
                    'current_end': Tensor[125],  # Position tracker
                    'cache': Tensor[125, 2, 8, 127, 256]  # KV cache
                },
                'transformer.layers.1.self_attn': { ... },
                ...  # 6 layers total for Pocket TTS
            }
            ```

            **Cache dimensions:**
            - Dim 0 (125): Batch size
            - Dim 1 (2): Keys and Values
            - Dim 2 (8): Number of attention heads
            - Dim 3 (127): Sequence length (audio frames)
            - Dim 4 (256): Embedding dimension per head

            ---

            ### Voice Blending

            **What it does:**
            ```python
            blended_cache = (1 - blend_ratio) * cache_A + blend_ratio * cache_B
            ```

            Weighted average of attention patterns from two voices.

            **Effect:** Combines voice characteristics like pitch, timbre, rhythm, breathiness.

            **Parameters exposed:**
            - `blend_ratio`: 0.0 (all A) to 1.0 (all B)

            ---

            ### Temperature Modification

            **What it does:**
            ```python
            mean = torch.nanmean(cache)
            modified = mean + (cache - mean) * temperature
            ```

            Scales variance around the mean.

            **Effect:** Controls how much the voice varies during speech.
            - `temperature > 1.0`: More varied/dynamic voice
            - `temperature < 1.0`: More consistent/monotone voice

            **Maps to:** Variance of attention weights in each layer's cache.

            ---

            ### Sharpness Modification

            **What it does:**
            ```python
            median = torch.nanmedian(cache)
            sign = torch.sign(cache - median)
            modified = cache + sharpness * sign * torch.abs(cache - median)
            ```

            Enhances or reduces contrast relative to median.

            **Effect:** Controls voice clarity and distinctiveness.
            - `sharpness > 0`: Crisper consonants, defined vowels
            - `sharpness < 0`: Softer, more muffled quality

            **Maps to:** Distribution of attention weights (peaked vs diffuse).

            ---

            ### Depth Modification

            **What it does:**
            ```python
            modified = cache + depth
            ```

            Shifts all cache values by a constant.

            **Effect:** Controls overall tonal quality.
            - `depth > 0`: Deeper, more resonant voice
            - `depth < 0`: Lighter, airier voice

            **Maps to:** Mean baseline of all attention values.

            ---

            ## Limitations

            1. **Not trained for this:** Model wasn't trained to handle modified KV caches
            2. **Heuristic:** These are educated guesses, not guarantees
            3. **Non-linear:** Combining modifications may have unexpected effects
            4. **Better alternatives:** Fine-tuning gives better production results

            See `FINE_TUNING_ANALYSIS.md` for training your own custom voice models.
            """)

        # Analysis tab
        with gr.Tab("🔬 Voice Analysis"):
            gr.Markdown("""
            ## Analyze Voice Characteristics

            Examine the internal representation (KV cache) of different voices.
            """)

            with gr.Row():
                analyze_voice = gr.Dropdown(
                    label="Select Voice to Analyze",
                    choices=PRESET_VOICES,
                    value="javert",
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

            analysis_output = gr.Markdown("Select a voice and click Analyze")

            analyze_btn.click(
                fn=lambda v: analyze_voice_state(get_voice_state(v)),
                inputs=[analyze_voice],
                outputs=[analysis_output],
            )

        # Comparison tab
        with gr.Tab("📊 Voice Comparison"):
            gr.Markdown("""
            ## Compare Two Voices

            See how similar or different two voices are in the embedding space.
            """)

            with gr.Row():
                compare_a = gr.Dropdown(
                    label="Voice A",
                    choices=PRESET_VOICES,
                    value="javert",
                )
                compare_b = gr.Dropdown(
                    label="Voice B",
                    choices=PRESET_VOICES,
                    value="marius",
                )

            compare_btn = gr.Button("Compare Voices", variant="primary")
            comparison_output = gr.Markdown("Select two voices and click Compare")

            compare_btn.click(
                fn=compare_voices,
                inputs=[compare_a, compare_b],
                outputs=[comparison_output],
            )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎛️  Starting Advanced Voice Mixer Studio")
    print("="*60)
    print("\nAvailable preset voices:")
    for voice in PRESET_VOICES:
        print(f"  • {voice}: {VOICE_INFO.get(voice, '')}")
    print("\n🌐 Launching web interface...")
    print("   Press Ctrl+C to stop.\n")

    demo.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=7860,
    )
