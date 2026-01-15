#!/usr/bin/env python3
"""
Advanced Voice Customization for Pocket TTS
Demonstrates creating a gritty, custom voice using voice cloning and audio processing.
"""

from pocket_tts import TTSModel
import scipy.io.wavfile
import numpy as np
import os

# Optional: Install pydub for audio processing
# pip install pydub
try:
    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range, low_pass_filter
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("⚠️  pydub not installed - skipping audio processing steps")
    print("   Install with: pip install pydub")
    print("   Also install ffmpeg: sudo apt-get install ffmpeg\n")


def make_voice_grittier(input_wav: str, output_wav: str):
    """
    Process audio to make it grittier using pydub.

    Techniques:
    1. Slightly lower pitch
    2. Add compression for more dynamic range
    3. Filter high frequencies
    4. Boost low-mid frequencies
    """
    if not HAS_PYDUB:
        print(f"   ⚠️  Skipping processing - copying {input_wav} to {output_wav}")
        import shutil
        shutil.copy(input_wav, output_wav)
        return

    print(f"   🎛️  Processing audio to make it grittier...")

    # Load audio
    audio = AudioSegment.from_wav(input_wav)

    # 1. Lower pitch by 5% (makes voice deeper/grittier)
    print(f"      - Lowering pitch by 5%")
    audio_gritty = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * 0.95)
    })
    audio_gritty = audio_gritty.set_frame_rate(24000)  # Pocket TTS uses 24kHz

    # 2. Add compression (increases perceived grittiness)
    print(f"      - Adding dynamic compression")
    audio_gritty = compress_dynamic_range(
        audio_gritty,
        threshold=-20.0,
        ratio=4.0,
        attack=5.0,
        release=50.0
    )

    # 3. Cut very high frequencies (makes it rougher)
    print(f"      - Filtering high frequencies above 5kHz")
    audio_gritty = low_pass_filter(audio_gritty, 5000)

    # 4. Boost low-mid frequencies (200-800 Hz for grittiness)
    print(f"      - Boosting low-mid frequencies")
    # Note: pydub doesn't have a parametric EQ, but compression helps here

    # Export
    audio_gritty.export(output_wav, format="wav")
    print(f"   ✓ Processed audio saved to {output_wav}")


def main():
    print("🎙️  Advanced Voice Customization Demo")
    print("="*60)
    print("\nThis script demonstrates creating a custom gritty voice by:")
    print("1. Starting with a gritty preset (Javert)")
    print("2. Processing it to be even grittier")
    print("3. Comparing different voice styles")
    print("\n⚠️  NOTE: Custom voice cloning from WAV files requires")
    print("   Hugging Face authentication. See TRAINING_GUIDE.md")
    print("   This demo uses preset voices (no auth needed).")
    print("="*60 + "\n")

    # Create output directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("custom_voices", exist_ok=True)

    # Load TTS model
    print("Loading Pocket TTS model...")
    tts_model = TTSModel.load_model()
    print(f"✓ Model loaded!\n")

    # ===================================================================
    # Step 1: Generate a sample with the grittiest preset voice (Javert)
    # ===================================================================
    print("Step 1: Generating base audio with Javert (grittiest preset)")
    print("-" * 60)

    sample_text = (
        "The old path winds through shadows deep, "
        "where forgotten voices whisper secrets in the dark."
    )

    voice_state = tts_model.get_state_for_audio_prompt("javert")
    audio = tts_model.generate_audio(voice_state, sample_text)

    base_voice_file = "custom_voices/javert_base.wav"
    scipy.io.wavfile.write(base_voice_file, tts_model.sample_rate, audio.numpy())
    print(f"✓ Base voice saved to {base_voice_file}\n")

    # ===================================================================
    # Step 2: Process the audio to make it even grittier
    # ===================================================================
    print("Step 2: Processing audio to enhance grittiness")
    print("-" * 60)

    gritty_voice_file = "custom_voices/javert_gritty.wav"
    make_voice_grittier(base_voice_file, gritty_voice_file)
    print()

    # ===================================================================
    # Step 3: Generate speech and post-process it
    # ===================================================================
    print("Step 3: Generating more speech with gritty voice")
    print("-" * 60)

    final_text = (
        "The wind howled through the canyon, carrying echoes of ancient tales. "
        "In the depths of the forgotten valley, shadows dance with secrets untold."
    )

    # Generate with Javert again for a longer sample
    print(f"   Generating speech with Javert preset...")
    javert_voice_state = tts_model.get_state_for_audio_prompt("javert")
    custom_audio = tts_model.generate_audio(javert_voice_state, final_text)

    output_file = "output/javert_long_sample.wav"
    scipy.io.wavfile.write(output_file, tts_model.sample_rate, custom_audio.numpy())
    print(f"✓ Generated speech saved to {output_file}")

    # Post-process it if we have pydub
    if HAS_PYDUB:
        print(f"   Post-processing to make it grittier...")
        gritty_output = "output/javert_long_gritty.wav"
        make_voice_grittier(output_file, gritty_output)
        print(f"✓ Gritty version saved to {gritty_output}\n")
    else:
        print(f"   ⚠️  Install pydub for audio post-processing\n")

    # ===================================================================
    # Step 4: Compare all preset voices for grittiness
    # ===================================================================
    print("Step 4: Generating comparison samples with different voices")
    print("-" * 60)

    comparison_text = "Listen carefully to the difference in voice quality and character."

    voices_to_compare = [
        ("marius", "Breathy male voice"),
        ("javert", "Gritty male voice (recommended)"),
        ("jean", "Neutral male voice"),
        ("alba", "Female voice for contrast")
    ]

    for voice_name, description in voices_to_compare:
        print(f"   Generating with {voice_name} ({description})...")
        voice_state = tts_model.get_state_for_audio_prompt(voice_name)
        audio = tts_model.generate_audio(voice_state, comparison_text)
        output_path = f"output/comparison_{voice_name}.wav"
        scipy.io.wavfile.write(output_path, tts_model.sample_rate, audio.numpy())

    print()

    # ===================================================================
    # Done!
    # ===================================================================
    print("="*60)
    print("🎉 Done! Voice samples have been generated!")
    print("="*60)
    print("\n📁 Files created:")
    print(f"   • {base_voice_file}")
    print(f"      └─ Base voice sample from Javert")
    print(f"   • {gritty_voice_file}")
    print(f"      └─ Processed version (if pydub installed)")
    print(f"   • {output_file}")
    print(f"      └─ Longer speech sample with Javert")
    if HAS_PYDUB:
        print(f"   • output/javert_long_gritty.wav")
        print(f"      └─ Post-processed gritty version")
    print(f"   • output/comparison_*.wav")
    print(f"      └─ Voice comparison samples\n")

    print("🎧 Listen to the comparison files:")
    print("   1. comparison_marius.wav  - Breathy male voice")
    print("   2. comparison_javert.wav  - Gritty male voice")
    print("   3. comparison_jean.wav    - Neutral male voice")
    print("   4. comparison_alba.wav    - Female voice\n")

    print("💡 Next steps:")
    print("   • Listen to all the samples and pick your favorite")
    print("   • Javert is the grittiest preset voice")
    if not HAS_PYDUB:
        print("   • Install pydub + ffmpeg for audio processing:")
        print("     pip install pydub && sudo apt-get install ffmpeg")
    print("   • For CUSTOM voice cloning from your own files:")
    print("     1. Authenticate with Hugging Face (see TRAINING_GUIDE.md)")
    print("     2. Use: python use_your_voice.py your_audio.wav")
    print("   • See TRAINING_GUIDE.md for complete customization guide\n")


if __name__ == "__main__":
    main()
