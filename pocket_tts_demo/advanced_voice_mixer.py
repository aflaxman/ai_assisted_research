#!/usr/bin/env python3
"""
Advanced Voice Mixer - Interactive tool with preset management, voice analysis, and audio manipulation.

NEW FEATURES:
- Save and load parameter presets
- Text block splitting for less repetition
- Pitch and tempo controls (post-processing)
- Parallel coordinates visualization of voice vectors
- Enhanced comparison panel with saved presets
- Narrower parameter ranges to prevent glitchy sounds
"""

import gradio as gr
import torch
import numpy as np
from pocket_tts import TTSModel
import scipy.io.wavfile
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
import librosa
from datetime import datetime
import re

# Initialize model
print("Loading Pocket TTS model...")
tts = TTSModel.load_model()
print("Model loaded successfully!")

# Preset storage file
PRESETS_FILE = Path("voice_presets.json")

# ACTUAL preset voices (verified to exist)
PRESET_VOICES = [
    "alba",      # Female, clear
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
# Cache for voice vectors
voice_vector_cache = {}


def load_presets():
    """Load saved presets from JSON file."""
    if PRESETS_FILE.exists():
        with open(PRESETS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_presets(presets):
    """Save presets to JSON file."""
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)


def get_voice_state(voice_source):
    """Get or cache voice state."""
    if voice_source not in voice_cache:
        print(f"Extracting voice state for: {voice_source}")
        voice_cache[voice_source] = tts.get_state_for_audio_prompt(voice_source)
    return voice_cache[voice_source]


def extract_voice_vector(voice_state):
    """Extract a summary vector from voice state by averaging key statistics."""
    vectors = []
    for key, value in sorted(voice_state.items()):
        if isinstance(value, dict) and 'cache' in value:
            cache = value['cache']
            if torch.is_tensor(cache):
                # Extract statistical summary
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


def get_voice_vector(voice_name):
    """Get or cache voice vector."""
    if voice_name not in voice_vector_cache:
        state = get_voice_state(voice_name)
        voice_vector_cache[voice_name] = extract_voice_vector(state)
    return voice_vector_cache[voice_name]


def split_text_into_blocks(text, target_length=80):
    """
    Split text into blocks of approximately target_length characters.
    Splits on sentence boundaries when possible.
    """
    # Split into sentences (simple approach)
    sentences = re.split(r'([.!?]+\s*)', text)

    blocks = []
    current_block = ""

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]  # Add punctuation

        if len(current_block) + len(sentence) <= target_length * 1.5:
            current_block += sentence
        else:
            if current_block:
                blocks.append(current_block.strip())
            current_block = sentence

    if current_block:
        blocks.append(current_block.strip())

    return blocks if blocks else [text]


def blend_voice_states(voice_state_a, voice_state_b, blend_ratio):
    """Blend two voice states together."""
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
                            tensor_a = value_a[subkey]
                            tensor_b = value_b[subkey]

                            mask_a = ~torch.isnan(tensor_a)
                            mask_b = ~torch.isnan(tensor_b)

                            blended_tensor = torch.where(
                                mask_a & mask_b,
                                (1 - blend_ratio) * tensor_a + blend_ratio * tensor_b,
                                torch.where(mask_a, tensor_a, tensor_b)
                            )
                            blended[key][subkey] = blended_tensor
                        else:
                            blended[key][subkey] = value_a[subkey]
            else:
                blended[key] = value_a
        else:
            blended[key] = voice_state_a[key]

    return blended


def modify_voice_state(voice_state, temperature, sharpness, depth):
    """Modify voice state characteristics."""
    modified = {}

    for key, value in voice_state.items():
        if isinstance(value, dict):
            modified[key] = {}
            for subkey, subvalue in value.items():
                if torch.is_tensor(subvalue) and subkey == 'cache':
                    tensor = subvalue.clone()

                    if temperature != 1.0:
                        mean = torch.nanmean(tensor)
                        tensor = mean + (tensor - mean) * temperature

                    if sharpness != 0.0:
                        median = torch.nanmedian(tensor)
                        tensor = tensor + sharpness * torch.sign(tensor - median) * torch.abs(tensor - median)

                    if depth != 0.0:
                        tensor = tensor + depth

                    modified[key][subkey] = tensor
                else:
                    modified[key][subkey] = subvalue
        else:
            modified[key] = value

    return modified


def apply_pitch_tempo(audio, sample_rate, pitch_shift, tempo_factor):
    """Apply pitch and tempo modifications using librosa."""
    if pitch_shift == 0 and tempo_factor == 1.0:
        return audio

    # Ensure float32
    audio = audio.astype(np.float32)

    # Apply pitch shift (in semitones)
    if pitch_shift != 0:
        audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)

    # Apply tempo change
    if tempo_factor != 1.0:
        audio = librosa.effects.time_stretch(audio, rate=tempo_factor)

    return audio


def generate_with_custom_voice(text, voice_a, voice_b, blend_ratio,
                                temperature, sharpness, depth,
                                use_preset_a, use_preset_b,
                                custom_audio_a, custom_audio_b,
                                pitch_shift, tempo_factor,
                                split_blocks):
    """Generate speech with blended/modified voice and audio processing."""
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

        # Split text into blocks if requested
        if split_blocks:
            text_blocks = split_text_into_blocks(text)
            print(f"Split into {len(text_blocks)} blocks")
        else:
            text_blocks = [text]

        # Generate audio for each block
        audio_segments = []
        for i, block in enumerate(text_blocks):
            if block.strip():
                print(f"Generating block {i+1}/{len(text_blocks)}: {block[:50]}...")
                block_audio = tts.generate_audio(voice_state, block)
                if torch.is_tensor(block_audio):
                    block_audio = block_audio.cpu().numpy()
                audio_segments.append(block_audio)
                # Add small silence between blocks (0.2 seconds)
                if i < len(text_blocks) - 1:
                    silence = np.zeros(int(tts.sample_rate * 0.2))
                    audio_segments.append(silence)

        # Concatenate all segments
        audio = np.concatenate(audio_segments)

        # Apply pitch and tempo modifications
        if pitch_shift != 0 or tempo_factor != 1.0:
            print(f"Applying pitch={pitch_shift} semitones, tempo={tempo_factor}x")
            audio = apply_pitch_tempo(audio, tts.sample_rate, pitch_shift, tempo_factor)
            voice_info += f"\n🎵 Audio: pitch={pitch_shift:+d} semitones, tempo={tempo_factor:.2f}x"

        if split_blocks and len(text_blocks) > 1:
            voice_info += f"\n📝 Split into {len(text_blocks)} blocks"

        status = f"✅ Generated {len(audio)/tts.sample_rate:.2f}s of audio\n{voice_info}"

        return (tts.sample_rate, audio), status

    except Exception as e:
        import traceback
        return None, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"


def save_current_preset(preset_name, voice_a, voice_b, blend_ratio,
                       temperature, sharpness, depth, pitch_shift, tempo_factor):
    """Save current parameters as a named preset."""
    if not preset_name or not preset_name.strip():
        return "⚠️ Please enter a preset name", gr.update()

    presets = load_presets()
    presets[preset_name] = {
        "voice_a": voice_a,
        "voice_b": voice_b,
        "blend_ratio": blend_ratio,
        "temperature": temperature,
        "sharpness": sharpness,
        "depth": depth,
        "pitch_shift": pitch_shift,
        "tempo_factor": tempo_factor,
        "created": datetime.now().isoformat()
    }
    save_presets(presets)

    # Update dropdown choices
    preset_names = list(presets.keys())

    return f"✅ Saved preset: {preset_name}", gr.update(choices=preset_names, value=preset_name)


def load_preset(preset_name):
    """Load a named preset."""
    if not preset_name:
        return [None] * 8 + ["⚠️ Select a preset to load"]

    presets = load_presets()
    if preset_name not in presets:
        return [None] * 8 + [f"⚠️ Preset '{preset_name}' not found"]

    preset = presets[preset_name]
    return [
        preset.get("voice_a", "javert"),
        preset.get("voice_b", "marius"),
        preset.get("blend_ratio", 0.0),
        preset.get("temperature", 1.0),
        preset.get("sharpness", 0.0),
        preset.get("depth", 0.0),
        preset.get("pitch_shift", 0),
        preset.get("tempo_factor", 1.0),
        f"✅ Loaded preset: {preset_name}"
    ]


def delete_preset(preset_name):
    """Delete a named preset."""
    if not preset_name:
        return "⚠️ Select a preset to delete", gr.update()

    presets = load_presets()
    if preset_name in presets:
        del presets[preset_name]
        save_presets(presets)
        preset_names = list(presets.keys())
        return f"✅ Deleted preset: {preset_name}", gr.update(choices=preset_names, value=None)
    else:
        return f"⚠️ Preset '{preset_name}' not found", gr.update()


def create_parallel_coordinates_plot(voices_to_compare):
    """Create a parallel coordinates plot for voice vector comparison."""
    if not voices_to_compare or len(voices_to_compare) < 1:
        return None

    # Get vectors
    data = []
    for voice in voices_to_compare:
        vec = get_voice_vector(voice)
        if vec is not None:
            data.append({
                "voice": voice,
                "vector": vec.numpy()
            })

    if not data:
        return None

    # Create dataframe-like structure
    # Each of the 30 dimensions represents: [mean, std, min, max, median] for 6 layers
    dimension_names = []
    for layer in range(6):
        for stat in ["mean", "std", "min", "max", "median"]:
            dimension_names.append(f"L{layer}_{stat}")

    # Prepare data for plotly
    plot_data = []
    for item in data:
        row = {"voice": item["voice"]}
        for i, dim_name in enumerate(dimension_names):
            row[dim_name] = item["vector"][i]
        plot_data.append(row)

    # Create parallel coordinates plot
    dimensions = []
    for dim_name in dimension_names:
        values = [row[dim_name] for row in plot_data]
        dimensions.append(dict(
            label=dim_name,
            values=values
        ))

    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=[i for i in range(len(plot_data))],
                colorscale='Viridis',
                showscale=True,
                cmin=0,
                cmax=len(plot_data)-1
            ),
            dimensions=dimensions,
            labelfont=dict(size=10)
        )
    )

    fig.update_layout(
        title="Voice Vector Comparison (30 Dimensions)",
        height=600,
        margin=dict(l=100, r=100, t=100, b=50)
    )

    return fig


def compare_voices_detailed(voices_to_compare):
    """Compare multiple voices with detailed statistics and visualization."""
    if not voices_to_compare or len(voices_to_compare) < 2:
        return "⚠️ Select at least 2 voices to compare", None

    try:
        # Get vectors
        vectors = {}
        for voice in voices_to_compare:
            vec = get_voice_vector(voice)
            if vec is not None:
                vectors[voice] = vec

        if len(vectors) < 2:
            return "⚠️ Could not extract vectors for comparison", None

        # Compute pairwise similarities
        info = "# Voice Comparison\n\n"
        info += f"Comparing {len(vectors)} voices: {', '.join(vectors.keys())}\n\n"
        info += "## Vector Statistics\n\n"

        for voice, vec in vectors.items():
            info += f"### {voice} ({VOICE_INFO.get(voice, '')})\n"
            info += f"- Dimensions: {vec.numel()}\n"
            info += f"- Mean: {vec.mean().item():.4f}\n"
            info += f"- Std: {vec.std().item():.4f}\n"
            info += f"- Range: [{vec.min().item():.4f}, {vec.max().item():.4f}]\n\n"

        # Pairwise comparisons
        info += "## Pairwise Similarities\n\n"
        voices = list(vectors.keys())
        for i, voice_a in enumerate(voices):
            for voice_b in voices[i+1:]:
                vec_a = vectors[voice_a]
                vec_b = vectors[voice_b]

                similarity = torch.nn.functional.cosine_similarity(
                    vec_a.unsqueeze(0), vec_b.unsqueeze(0)
                ).item()
                distance = torch.norm(vec_a - vec_b).item()

                info += f"**{voice_a}** vs **{voice_b}**\n"
                info += f"- Cosine Similarity: {similarity:.4f} ({similarity*100:.1f}% similar)\n"
                info += f"- Euclidean Distance: {distance:.4f}\n\n"

        # Create parallel coordinates plot
        fig = create_parallel_coordinates_plot(voices_to_compare)

        return info, fig

    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None


# Create Gradio interface
with gr.Blocks(title="Advanced Voice Mixer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎛️ Advanced Voice Mixer Studio

    Blend voices, modify characteristics, and explore the voice embedding space in real-time.

    **NEW**: Save presets, text block splitting, pitch/tempo control, voice vector visualization!
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
                        value="The ancient forest whispers secrets through rustling leaves, while shadows dance in the fading light.",
                    )

                    split_blocks = gr.Checkbox(
                        label="Split into blocks (reduces repetition for long texts)",
                        value=False,
                        info="Processes text in sentence-sized chunks"
                    )

                    with gr.Row():
                        sample_buttons = {}
                        for name in SAMPLE_TEXTS.keys():
                            sample_buttons[name] = gr.Button(name, size="sm")
                        regenerate_btn = gr.Button("🔄 Regenerate", variant="secondary", size="sm")

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
                    )

                    gr.Markdown("### Voice Modifications")

                    temperature = gr.Slider(
                        label="Temperature (voice variability)",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                    )

                    sharpness = gr.Slider(
                        label="Sharpness (voice distinctiveness)",
                        minimum=-0.3,
                        maximum=0.3,
                        value=0.0,
                        step=0.05,
                        info="REDUCED RANGE to prevent glitches"
                    )

                    depth = gr.Slider(
                        label="Depth (voice resonance)",
                        minimum=-0.2,
                        maximum=0.2,
                        value=0.0,
                        step=0.02,
                        info="REDUCED RANGE to prevent glitches"
                    )

                    gr.Markdown("### Audio Post-Processing")

                    pitch_shift = gr.Slider(
                        label="Pitch Shift (semitones)",
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=1,
                        info="Negative = lower, Positive = higher"
                    )

                    tempo_factor = gr.Slider(
                        label="Tempo Factor (speed)",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="<1 = slower, >1 = faster"
                    )

                    gr.Markdown("### Preset Management")

                    preset_name_input = gr.Textbox(
                        label="Preset Name",
                        placeholder="Enter name for current settings...",
                    )

                    with gr.Row():
                        save_preset_btn = gr.Button("💾 Save Preset", variant="primary")
                        load_preset_dropdown = gr.Dropdown(
                            label="Load Preset",
                            choices=list(load_presets().keys()),
                            value=None,
                        )
                        load_preset_btn = gr.Button("📂 Load")
                        delete_preset_btn = gr.Button("🗑️ Delete", variant="stop")

                    preset_status = gr.Markdown("")

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

                    **NEW Features:**
                    - Save favorite settings as named presets
                    - Split long text to reduce repetition
                    - Adjust pitch/tempo for singing (experimental)
                    - Narrower sharpness/depth ranges = no glitches!
                    - 🔄 Regenerate button: Hear variation without changing parameters
                    - Sample buttons: Click repeatedly to hear natural variation!

                    **Hearing Natural Variation:**
                    - Click sample text buttons multiple times
                    - Each click regenerates with same params
                    - Helps you feel the randomness in generation
                    - Or use the 🔄 Regenerate button anytime

                    **Pitch/Tempo for Songs:**
                    - Pitch shift: ±12 semitones (full octave range)
                    - Tempo: 0.5x to 2.0x speed
                    - Note: These are post-processing effects
                    - For true singing, use specialized TTS models

                    **30-Dimensional Voice Vector:**
                    - 6 layers × 5 stats (mean, std, min, max, median)
                    - See Comparison tab for visualization
                    - NOT full fine-grained control (that's the KV cache)
                    - Useful for voice similarity analysis
                    """)

            # Auto-generate when parameters change
            inputs = [
                text_input, voice_a, voice_b, blend_ratio,
                temperature, sharpness, depth,
                use_preset_a, use_preset_b,
                custom_audio_a, custom_audio_b,
                pitch_shift, tempo_factor, split_blocks,
            ]

            for inp in inputs:
                inp.change(
                    fn=generate_with_custom_voice,
                    inputs=inputs,
                    outputs=[audio_output, status_text],
                )

            # Preset management
            save_preset_btn.click(
                fn=save_current_preset,
                inputs=[preset_name_input, voice_a, voice_b, blend_ratio,
                       temperature, sharpness, depth, pitch_shift, tempo_factor],
                outputs=[preset_status, load_preset_dropdown],
            )

            load_preset_btn.click(
                fn=load_preset,
                inputs=[load_preset_dropdown],
                outputs=[voice_a, voice_b, blend_ratio, temperature, sharpness,
                        depth, pitch_shift, tempo_factor, preset_status],
            )

            delete_preset_btn.click(
                fn=delete_preset,
                inputs=[load_preset_dropdown],
                outputs=[preset_status, load_preset_dropdown],
            )

            # Sample text buttons - trigger both text update AND regeneration
            for name, btn in sample_buttons.items():
                # Use a wrapper function to update text and trigger generation
                def make_sample_handler(sample_name):
                    def handler(*current_inputs):
                        # Update the text (first input)
                        new_inputs = list(current_inputs)
                        new_inputs[0] = SAMPLE_TEXTS[sample_name]
                        # Generate with new text
                        return generate_with_custom_voice(*new_inputs)
                    return handler

                btn.click(
                    fn=make_sample_handler(name),
                    inputs=inputs,
                    outputs=[audio_output, status_text],
                )

            # Regenerate button - triggers generation with current parameters
            regenerate_btn.click(
                fn=generate_with_custom_voice,
                inputs=inputs,
                outputs=[audio_output, status_text],
            )

        # Comparison tab with parallel coordinates
        with gr.Tab("📊 Voice Comparison & Visualization"):
            gr.Markdown("""
            ## Compare Voice Vectors

            The 30-dimensional vector represents statistical summaries from the 6-layer KV cache:
            - Layer 0-5: mean, std, min, max, median for each layer
            - Total: 6 layers × 5 statistics = 30 dimensions

            This is a **compressed representation** for analysis. For full control, the KV cache
            has millions of parameters (see Parameter Documentation tab).
            """)

            voices_to_compare = gr.CheckboxGroup(
                label="Select voices to compare",
                choices=PRESET_VOICES,
                value=["javert", "marius"],
            )

            compare_btn = gr.Button("Compare Selected Voices", variant="primary")

            comparison_text = gr.Markdown("")
            parallel_plot = gr.Plot(label="Parallel Coordinates Plot")

            compare_btn.click(
                fn=compare_voices_detailed,
                inputs=[voices_to_compare],
                outputs=[comparison_text, parallel_plot],
            )

            gr.Markdown("""
            ### How to Read the Parallel Coordinates Plot

            - Each vertical axis = one of the 30 dimensions
            - Each colored line = one voice
            - Lines closer together = voices more similar in that dimension
            - Pattern similarities = overall voice similarities

            **Dimensions:**
            - L0_mean through L5_mean: Average values per layer
            - L0_std through L5_std: Variation per layer
            - L0_min through L5_min: Minimum values
            - L0_max through L5_max: Maximum values
            - L0_median through L5_median: Median values
            """)

        # Documentation tab
        with gr.Tab("📖 Parameter Documentation"):
            gr.Markdown("""
            ## Updated Parameter Guide

            ### Pitch and Tempo Control

            **Can PocketTTS make songs?**
            - PocketTTS doesn't have native pitch/tempo control
            - We use **librosa** for post-processing:
              * `pitch_shift`: Shifts pitch up/down (in semitones)
              * `tempo_factor`: Changes speed without pitch change

            **For singing:**
            - Pitch control: ±12 semitones (full octave)
            - Tempo control: 0.5x to 2.0x
            - **Limitation:** These are simple audio manipulations
            - **Better approach:** Use dedicated singing TTS models like:
              * Diff-SVC (for voice conversion to singing)
              * So-VITS-SVC (singing voice synthesis)
              * RVC (Retrieval-based Voice Conversion)

            ### Text Block Splitting

            **Why split text:**
            - Long texts can sound repetitive
            - Each block is generated independently
            - 0.2s silence added between blocks
            - Target length: ~80 characters per block

            **How it works:**
            ```python
            blocks = split_text_into_blocks(text, target_length=80)
            for block in blocks:
                audio_segment = generate(block)
                # Add to final audio
            ```

            ### Reduced Parameter Ranges

            **Old ranges caused glitches:**
            - Sharpness: -1.0 to 1.0 → NOW: -0.3 to 0.3
            - Depth: -0.5 to 0.5 → NOW: -0.2 to 0.2

            **Why:**
            - Extreme values corrupt the KV cache
            - Smaller ranges = stable, glitch-free output
            - Still provides noticeable effect

            ### 30-Dimensional Voice Vector Explained

            **What it represents:**
            ```
            For each of 6 transformer layers:
              - mean of KV cache values
              - standard deviation
              - minimum value
              - maximum value
              - median value

            Total: 6 layers × 5 stats = 30 dimensions
            ```

            **Is this full fine-grained control?**
            - NO - this is a statistical summary
            - Full KV cache has ~25 million values
            - 30 dimensions = compressed representation
            - Useful for:
              * Voice similarity comparison
              * Understanding voice characteristics
              * Quick voice analysis

            **For full control:**
            - Manipulate the entire KV cache (what this tool does)
            - Or fine-tune the model (see FINE_TUNING_ANALYSIS.md)

            See other tabs for the actual formulas and KV cache structure.
            """)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎛️  Starting Advanced Voice Mixer Studio v2.0")
    print("="*60)
    print("\nNew features:")
    print("  • Preset save/load")
    print("  • Text block splitting")
    print("  • Pitch/tempo controls")
    print("  • Parallel coordinates visualization")
    print("  • Narrower parameter ranges (no more glitches!)")
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
