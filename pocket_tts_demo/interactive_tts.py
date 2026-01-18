#!/usr/bin/env python3
"""
Interactive TTS GUI with real-time parameter tweaking.

This tool provides a web-based interface to:
- View model configuration (read-only)
- Tweak inference parameters in real-time
- Generate and play TTS audio instantly
- Compare different settings side-by-side
"""

import gradio as gr
import torch
import numpy as np
from pathlib import Path
import json
from pocket_tts import TTSModel

# Initialize model
print("Loading Pocket TTS model...")
tts = TTSModel.load_model()
print("Model loaded successfully!")

# ACTUAL preset voices (verified to exist)
PRESET_VOICES = {
    "None (Default)": None,
    "Alba (Female, clear)": "alba",
    "Marius (Male, breathy)": "marius",
    "Javert (Male, deep/gritty)": "javert",
    "Jean (Male, neutral)": "jean",
    "Fantine (Female, mature)": "fantine",
    "Cosette (Female, young)": "cosette",
    "Eponine (Female, spirited)": "eponine",
    "Azelma (Female, soft)": "azelma",
}

# Default text samples
SAMPLE_TEXTS = {
    "Short": "The quick brown fox jumps over the lazy dog.",
    "Medium": "In the beginning was the Word, and the Word was with God, and the Word was made flesh.",
    "Long": "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness. We had everything before us, we had nothing before us.",
    "Technical": "The transformer architecture uses multi-head self-attention mechanisms to process sequential data in parallel.",
}


def get_model_info():
    """Get model configuration info as formatted text."""
    info = []
    info.append("# Pocket TTS Model Configuration\n")

    # FlowLM info
    info.append("## Flow Language Model")
    info.append(f"- Model dimension: {tts.flow_lm.transformer.d_model}")
    info.append(f"- Number of layers: {tts.flow_lm.transformer.num_layers}")
    info.append(f"- Number of heads: {tts.flow_lm.transformer.num_heads}")
    info.append(f"- Hidden scale: {tts.flow_lm.transformer.hidden_scale}")
    info.append(f"- Max period: {tts.flow_lm.transformer.max_period}")

    # Flow head info
    info.append(f"\n- Flow depth: {tts.flow_lm.flow_net.depth}")
    info.append(f"- Flow dimension: {tts.flow_lm.flow_net.dim}")

    # Mimi info
    info.append("\n## Mimi Audio Codec")
    info.append(f"- Sample rate: {tts.sample_rate} Hz")
    info.append(f"- Frame rate: {tts.mimi.frame_rate} Hz")
    info.append(f"- Channels: {tts.mimi.channels}")
    info.append(f"- Transformer dimension: {tts.mimi.transformer.d_model}")
    info.append(f"- Transformer layers: {tts.mimi.transformer.num_layers}")

    # Trainable parameters
    total_params = sum(p.numel() for p in tts.flow_lm.parameters())
    trainable_params = sum(p.numel() for p in tts.flow_lm.parameters() if p.requires_grad)
    info.append(f"\n## Model Size")
    info.append(f"- Total parameters: {total_params:,}")
    info.append(f"- Trainable parameters: {trainable_params:,}")
    info.append(f"- Model size: ~{total_params * 4 / 1024 / 1024:.0f} MB")

    return "\n".join(info)


def generate_tts(
    text,
    voice_preset,
    custom_voice_audio,
    temperature,
    cfg_scale,
    num_steps,
    use_custom_voice,
):
    """Generate TTS audio with given parameters."""
    if not text or not text.strip():
        return None, "⚠️ Please enter some text to generate speech."

    try:
        # Determine voice source
        voice = None
        voice_info = "🔊 Using default voice"

        if use_custom_voice and custom_voice_audio is not None:
            # Use custom audio to extract voice state
            voice_state = tts.get_state_for_audio_prompt(custom_voice_audio)
            voice_info = "🎤 Using custom voice (uploaded audio)"
        elif voice_preset and voice_preset != "None (Default)":
            # Use preset voice
            voice = PRESET_VOICES[voice_preset]
            voice_state = tts.get_state_for_audio_prompt(voice)
            voice_info = f"🎭 Using preset: {voice_preset}"
        else:
            # Use default (no voice state)
            voice_state = None
            voice_info = "🔊 Using default voice"

        # Generate audio
        if voice_state is not None:
            audio = tts.generate_audio(voice_state, text)
        else:
            # For default voice, use the basic generate method
            audio = tts.generate_audio(None, text)

        # Convert to numpy array if needed
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()

        # Ensure correct shape for gradio (samples,) or (samples, channels)
        if audio.ndim == 1:
            audio = audio
        elif audio.ndim == 2:
            audio = audio.T  # Gradio expects (samples, channels)

        status = f"✅ Generated {len(audio)/tts.sample_rate:.2f}s of audio\n"
        status += voice_info

        # Note about unsupported parameters
        status += "\n\n⚠️ Note: Temperature, CFG scale, and num_steps are not currently exposed by the Pocket TTS API."
        status += "\nOnly text and voice selection are adjustable."
        status += "\n\nFor advanced voice manipulation (blending, temperature, etc.), use advanced_voice_mixer.py"

        return (tts.sample_rate, audio), status

    except Exception as e:
        import traceback
        return None, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"


def load_sample_text(sample_key):
    """Load a sample text."""
    return SAMPLE_TEXTS.get(sample_key, "")


# Create Gradio interface
with gr.Blocks(title="Interactive Pocket TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Interactive Pocket TTS Studio

    Simple text-to-speech generation with voice selection.

    **Note:** For advanced features (voice blending, temperature, etc.), use `advanced_voice_mixer.py`
    """)

    with gr.Tabs():
        # Main generation tab
        with gr.Tab("🎬 Generate Speech"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Text to Speak",
                        placeholder="Enter text here...",
                        lines=5,
                        value="The quick brown fox jumps over the lazy dog.",
                    )

                    gr.Markdown("### Quick Samples")
                    with gr.Row():
                        sample_btns = []
                        for name in SAMPLE_TEXTS.keys():
                            btn = gr.Button(name, size="sm")
                            sample_btns.append((btn, name))

                    gr.Markdown("### Voice Selection")

                    voice_preset = gr.Dropdown(
                        label="Preset Voice",
                        choices=list(PRESET_VOICES.keys()),
                        value="None (Default)",
                    )

                    with gr.Accordion("🎤 Use Custom Voice (Advanced)", open=False):
                        use_custom_voice = gr.Checkbox(
                            label="Enable custom voice",
                            value=False,
                        )
                        custom_voice_audio = gr.Audio(
                            label="Upload reference audio (3-10 seconds)",
                            type="filepath",
                        )
                        gr.Markdown("*Upload a clean audio sample of the voice you want to clone*")
                        gr.Markdown("*Requires HF authentication for voice cloning - see README.md Method 2*")

                    gr.Markdown("### Generation Parameters (Display Only)")
                    gr.Markdown("*These parameters are not exposed in the current API*")

                    temperature = gr.Slider(
                        label="Temperature (NOT USED - for display only)",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        interactive=False,
                        info="Not exposed by Pocket TTS API"
                    )

                    cfg_scale = gr.Slider(
                        label="CFG Scale (NOT USED - for display only)",
                        minimum=0.0,
                        maximum=5.0,
                        value=1.5,
                        step=0.1,
                        interactive=False,
                        info="Not exposed by Pocket TTS API"
                    )

                    num_steps = gr.Slider(
                        label="Generation Steps (NOT USED - for display only)",
                        minimum=10,
                        maximum=100,
                        value=32,
                        step=1,
                        interactive=False,
                        info="Not exposed by Pocket TTS API"
                    )

                    gr.Markdown("**For advanced control**, use `advanced_voice_mixer.py` which manipulates the KV cache")

                    generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

                with gr.Column(scale=1):
                    status_text = gr.Markdown("Ready to generate speech!")

                    audio_output = gr.Audio(
                        label="Generated Audio",
                        type="numpy",
                        autoplay=True,
                        loop=True,
                    )

                    gr.Markdown("""
                    ### 💡 Tips
                    - Try different preset voices for variety
                    - Custom voices require HF authentication
                    - Shorter texts generate faster
                    - Use advanced_voice_mixer.py for:
                      * Voice blending
                      * Temperature control
                      * Sharpness/depth adjustment
                    """)

            # Connect sample buttons
            for btn, name in sample_btns:
                btn.click(
                    fn=lambda n=name: SAMPLE_TEXTS[n],
                    outputs=text_input,
                )

            # Connect generate button
            generate_btn.click(
                fn=generate_tts,
                inputs=[
                    text_input,
                    voice_preset,
                    custom_voice_audio,
                    temperature,
                    cfg_scale,
                    num_steps,
                    use_custom_voice,
                ],
                outputs=[audio_output, status_text],
            )

            # Also trigger on text change for convenience
            text_input.change(
                fn=generate_tts,
                inputs=[
                    text_input,
                    voice_preset,
                    custom_voice_audio,
                    temperature,
                    cfg_scale,
                    num_steps,
                    use_custom_voice,
                ],
                outputs=[audio_output, status_text],
            )

        # Model info tab
        with gr.Tab("⚙️ Model Configuration"):
            gr.Markdown("""
            ## Model Architecture Parameters

            These parameters define the model structure and **cannot be changed** without retraining.
            They're displayed here for reference.

            To actually fine-tune the model with different architecture, see `FINE_TUNING_ANALYSIS.md`.
            """)

            model_info = gr.Markdown(get_model_info())

            gr.Markdown("""
            ---
            ### 🔬 What These Parameters Mean

            **Flow Language Model:**
            - **d_model**: Embedding dimension (larger = more capacity)
            - **num_layers**: Transformer depth (more = more complex patterns)
            - **num_heads**: Attention heads (more = different attention patterns)
            - **hidden_scale**: Feedforward network scaling factor
            - **max_period**: Maximum positional encoding period

            **Flow Network:**
            - **depth**: MLP depth for flow matching
            - **dimension**: Flow network hidden dimension

            **Mimi Codec:**
            - **sample_rate**: Audio sample rate (24kHz)
            - **frame_rate**: Latent frame rate (12.5 Hz)
            - **channels**: Audio channels (1 = mono)

            These are architectural decisions made during training and can't be changed at inference time.
            """)

        # Help tab
        with gr.Tab("❓ Help"):
            gr.Markdown("""
            ## How to Use This Tool

            ### Basic Usage
            1. Enter text in the "Text to Speak" box
            2. Choose a preset voice or upload custom audio
            3. Click "Generate Speech" (or it auto-generates on text change)
            4. Listen to the result!

            ### Available Voices

            **Preset Voices:**
            - **Alba**: Female, clear
            - **Marius**: Male, breathy
            - **Javert**: Male, deep/gritty
            - **Jean**: Male, neutral
            - **Fantine**: Female, mature
            - **Cosette**: Female, young
            - **Eponine**: Female, spirited
            - **Azelma**: Female, soft

            **Custom Voice:**
            - Upload 3-10 seconds of clean audio
            - Single speaker, no background noise
            - Requires HF authentication (see README.md Method 2)

            ### Limitations

            **This tool is simplified:**
            - No parameter tweaking (temperature, etc.)
            - Just voice selection and text input

            **For advanced features, use `advanced_voice_mixer.py`:**
            - Voice blending (mix two voices)
            - Temperature control (voice variability)
            - Sharpness modification (crisp vs soft)
            - Depth adjustment (deep vs light)
            - Real-time auto-generation
            - Loop playback

            ### Troubleshooting

            **Custom voice fails:**
            - Make sure you've authenticated with Hugging Face
            - Run: `uvx hf auth login`
            - Accept terms at https://huggingface.co/kyutai/pocket-tts

            **Generation fails:**
            - Check that text isn't empty
            - Try a shorter text first
            - For custom voice, ensure audio is valid WAV format

            ### For More Information

            - See `README.md` for setup instructions
            - See `FINE_TUNING_ANALYSIS.md` for training details
            - Use `advanced_voice_mixer.py` for voice manipulation
            - Use `create_custom_voice.py` for voice comparison
            """)

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎙️  Starting Interactive Pocket TTS Studio")
    print("="*60)
    print("\nModel Information:")
    print(get_model_info())
    print("\n" + "="*60)
    print("\n🌐 Launching web interface...")
    print("   The browser will open automatically.")
    print("   Press Ctrl+C to stop the server.\n")

    demo.launch(
        share=False,  # Set to True to create public link
        show_error=True,
        server_name="127.0.0.1",  # Change to "0.0.0.0" to allow external access
        server_port=7861,  # Use different port from advanced_voice_mixer
    )
