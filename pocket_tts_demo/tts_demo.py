#!/usr/bin/env python3
"""
Simple Pocket TTS Demo
Generates speech with different voices to find breathy, gritty tones.
"""

from pocket_tts import TTSModel
import scipy.io.wavfile
import os

def main():
    print("🎙️  Pocket TTS Demo - Loading model...")
    print("(This may take a moment on first run)")

    # Load the TTS model (keep in memory for multiple generations)
    tts_model = TTSModel.load_model()
    print(f"✓ Model loaded! Sample rate: {tts_model.sample_rate} Hz\n")

    # Text to speak - something that shows off voice character
    text = "The wind howled through the canyon, a low whisper of secrets long forgotten."

    # List of voices to try - some may have more breathy/gritty qualities
    voices = ["alba", "marius", "javert", "jean"]

    print(f"Generating speech: '{text}'\n")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Generate audio for each voice
    for voice in voices:
        print(f"🔊 Generating with voice: {voice}")

        # Get voice state from preset (just pass the voice name)
        voice_state = tts_model.get_state_for_audio_prompt(voice)

        # Generate audio
        audio = tts_model.generate_audio(voice_state, text)

        # Save to file
        output_file = f"output/{voice}_demo.wav"
        scipy.io.wavfile.write(output_file, tts_model.sample_rate, audio.numpy())
        print(f"   ✓ Saved to {output_file}")

    print("\n" + "="*60)
    print("🎉 Done! Check the 'output/' folder for generated audio files.")
    print("\nVoice Recommendations:")
    print("  • 'javert' - Often has a deeper, grittier tone")
    print("  • 'marius' - Can have breathy qualities")
    print("\nListen to each and pick your favorite!")
    print("="*60)

if __name__ == "__main__":
    main()
