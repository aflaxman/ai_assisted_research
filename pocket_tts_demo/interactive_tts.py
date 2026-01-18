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
from pocket_tts import PocketTTS

# Initialize model
print("Loading Pocket TTS model...")
tts = PocketTTS()
print("Model loaded successfully!")

# Get available preset voices
PRESET_VOICES = {
    "None (Default)": None,
    "Javert (Deep, commanding)": "javert",
    "Daisy (Light, youthful)": "daisy",
    "Whisperer (Soft, intimate)": "whisperer",
    "Narrator (Clear, authoritative)": "narrator",
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
    info.append(f"- Sample rate: {tts.mimi.sample_rate} Hz")
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
            # Use custom audio
            voice = custom_voice_audio
            voice_info = "🎤 Using custom voice (uploaded audio)"
        elif voice_preset and voice_preset != "None (Default)":
            # Use preset voice
            voice = PRESET_VOICES[voice_preset]
            voice_info = f"🎭 Using preset: {voice_preset}"

        # Generate audio
        # Note: Not all parameters may be exposed by pocket-tts API
        # We'll use what's available
        gen_params = {}

        # Check what parameters the generate method accepts
        import inspect
        sig = inspect.signature(tts.generate)

        # Add parameters if they're supported
        if 'temperature' in sig.parameters:
            gen_params['temperature'] = temperature
        if 'cfg_scale' in sig.parameters:
            gen_params['cfg_scale'] = cfg_scale
        if 'num_steps' in sig.parameters:
            gen_params['num_steps'] = num_steps

        # Generate
        audio = tts.generate(text, voice=voice, **gen_params)

        # Convert to numpy array if needed
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()

        # Ensure correct shape for gradio (samples,) or (samples, channels)
        if audio.ndim == 1:
            audio = audio
        elif audio.ndim == 2:
            audio = audio.T  # Gradio expects (samples, channels)

        status = f"✅ Generated {len(audio)/tts.mimi.sample_rate:.2f}s of audio\n"
        status += voice_info

        # Add warning if parameters aren't supported
        unsupported = []
        if 'temperature' not in sig.parameters:
            unsupported.append('temperature')
        if 'cfg_scale' not in sig.parameters:
            unsupported.append('cfg_scale')
        if 'num_steps' not in sig.parameters:
            unsupported.append('num_steps')

        if unsupported:
            status += f"\n\n⚠️ Note: These parameters aren't supported by the current API: {', '.join(unsupported)}"
            status += "\nOnly text and voice selection are currently adjustable."

        return (tts.mimi.sample_rate, audio), status

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def load_sample_text(sample_key):
    """Load a sample text."""
    return SAMPLE_TEXTS.get(sample_key, "")


# Create Gradio interface
with gr.Blocks(title="Interactive Pocket TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Interactive Pocket TTS Studio

    Real-time text-to-speech generation with adjustable parameters.

    **Note:** Most config parameters are model architecture settings (fixed at training time).
    This tool exposes the **inference parameters** you can adjust in real-time.
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

                    gr.Markdown("### Generation Parameters")
                    gr.Markdown("*Note: These parameters may not be exposed in the current API*")

                    temperature = gr.Slider(
                        label="Temperature (controls randomness)",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        info="Higher = more varied, Lower = more deterministic"
                    )

                    cfg_scale = gr.Slider(
                        label="Classifier-Free Guidance Scale",
                        minimum=0.0,
                        maximum=5.0,
                        value=1.5,
                        step=0.1,
                        info="Higher = stronger text conditioning"
                    )

                    num_steps = gr.Slider(
                        label="Number of Generation Steps",
                        minimum=10,
                        maximum=100,
                        value=32,
                        step=1,
                        info="More steps = higher quality but slower"
                    )

                    generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

                with gr.Column(scale=1):
                    status_text = gr.Markdown("Ready to generate speech!")

                    audio_output = gr.Audio(
                        label="Generated Audio",
                        type="numpy",
                        autoplay=False,
                    )

                    gr.Markdown("""
                    ### 💡 Tips
                    - Start with preset voices for consistent results
                    - Custom voices require 3-10s of clear audio
                    - Shorter texts generate faster
                    - Temperature affects voice variation
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
            3. Click "Generate Speech"
            4. Listen to the result!

            ### Parameter Explanation

            **Adjustable at Inference:**
            - **Text**: What to say (obviously!)
            - **Voice**: Which voice to use (preset or custom)
            - **Temperature/CFG/Steps**: May not be exposed by current API

            **Fixed at Training:**
            - All model architecture parameters (see Configuration tab)
            - These define the model structure and can't change

            ### Voice Options

            **Preset Voices:**
            - **Javert**: Deep, commanding, authoritative
            - **Daisy**: Light, youthful, energetic
            - **Whisperer**: Soft, intimate, quiet
            - **Narrator**: Clear, neutral, professional

            **Custom Voice:**
            - Upload 3-10 seconds of clean audio
            - Single speaker, no background noise
            - The model will clone the voice characteristics

            ### Troubleshooting

            **"Parameters not supported" warning:**
            - The Pocket TTS API may not expose all parameters
            - This is normal - focus on text and voice selection

            **Generation fails:**
            - Check that text isn't empty
            - For custom voice, ensure audio is valid
            - Try a shorter text first

            ### For More Information

            - See `README.md` for setup instructions
            - See `FINE_TUNING_ANALYSIS.md` for training details
            - See `create_custom_voice.py` for voice comparison tool
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
        server_port=7860,
    )
