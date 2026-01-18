#!/usr/bin/env python3
"""
Advanced Voice Mixer - Interactive tool to blend and manipulate voice characteristics.

This tool extracts voice embeddings from audio files and allows you to:
- Blend multiple voices together
- Interpolate between different voice characteristics
- Modify KV cache values to tweak voice properties
- Visualize voice embeddings
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

# Preset voices
PRESET_VOICES = ["javert", "daisy", "whisperer", "narrator"]

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
            source_a = f"Preset: {voice_a}"
        elif custom_audio_a is not None:
            state_a = tts.get_state_for_audio_prompt(custom_audio_a)
            source_a = "Custom audio A"
        else:
            return None, "⚠️ Please select Voice A"

        # Check if we're blending
        if blend_ratio > 0:
            if use_preset_b and voice_b:
                state_b = get_voice_state(voice_b)
                source_b = f"Preset: {voice_b}"
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


# Create Gradio interface
with gr.Blocks(title="Advanced Voice Mixer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎛️ Advanced Voice Mixer Studio

    Blend voices, modify characteristics, and explore the voice embedding space.

    **How it works**: Voice is represented as transformer attention cache (KV cache).
    This tool lets you blend these caches and modify their statistics.
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
                        lines=5,
                        value="The ancient forest whispers secrets through rustling leaves.",
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
                                value="daisy",
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

                    generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

                with gr.Column():
                    status_text = gr.Markdown("Ready to mix voices!")

                    audio_output = gr.Audio(
                        label="Generated Audio",
                        type="numpy",
                    )

                    gr.Markdown("""
                    ### 💡 Tips

                    **Blending:**
                    - Start with blend=0 to hear Voice A pure
                    - Gradually increase to hear the blend
                    - blend=1.0 is Voice B pure

                    **Modifications:**
                    - Temperature: affects variation (1.0 = original)
                    - Sharpness: affects clarity (0 = original)
                    - Depth: affects tone (0 = original)

                    **Experimentation:**
                    - Try blending opposite voices (javert + daisy)
                    - Combine blending with modifications
                    - Use custom audio for unique voices
                    """)

            generate_btn.click(
                fn=generate_with_custom_voice,
                inputs=[
                    text_input,
                    voice_a, voice_b, blend_ratio,
                    temperature, sharpness, depth,
                    use_preset_a, use_preset_b,
                    custom_audio_a, custom_audio_b,
                ],
                outputs=[audio_output, status_text],
            )

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
                    value="daisy",
                )

            compare_btn = gr.Button("Compare Voices", variant="primary")
            comparison_output = gr.Markdown("Select two voices and click Compare")

            compare_btn.click(
                fn=compare_voices,
                inputs=[compare_a, compare_b],
                outputs=[comparison_output],
            )

        # Help tab
        with gr.Tab("❓ Help"):
            gr.Markdown("""
            ## Understanding Voice Representation

            ### What is Voice State?

            When you provide audio (either preset or custom), the model extracts a **voice state**
            which is stored as the transformer's **key-value (KV) cache**. This cache contains:

            - Attention patterns from each transformer layer
            - Encoded voice characteristics (pitch, timbre, rhythm, etc.)
            - Sequential dependencies from the audio

            The KV cache is a high-dimensional representation (thousands of values per layer)
            that captures the essence of the voice.

            ### How Blending Works

            When you blend two voices with ratio `r`:
            ```
            blended_cache = (1 - r) * cache_A + r * cache_B
            ```

            This creates a weighted average of the attention patterns, effectively
            mixing the voice characteristics.

            ### How Modifications Work

            **Temperature**: Scales the variance around the mean
            - Higher → more variation in attention patterns
            - Lower → more consistent attention patterns

            **Sharpness**: Enhances contrast in the cache values
            - Positive → amplifies differences from median
            - Negative → softens differences

            **Depth**: Shifts the overall mean of cache values
            - Positive → deeper/darker tone
            - Negative → lighter/brighter tone

            ### Limitations

            - Modifications are heuristic (not guaranteed to work perfectly)
            - Extreme values may produce artifacts
            - The model wasn't trained for this type of manipulation
            - Best results come from subtle adjustments

            ### Advanced Usage

            For true voice customization, consider:
            1. Recording high-quality custom audio (10-30 seconds)
            2. Fine-tuning the model (see `FINE_TUNING_ANALYSIS.md`)
            3. Training on domain-specific speech data

            This tool is for experimentation and exploration!
            """)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎛️  Starting Advanced Voice Mixer Studio")
    print("="*60)
    print("\n🌐 Launching web interface...")
    print("   Press Ctrl+C to stop.\n")

    demo.launch(
        share=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=7860,
    )
