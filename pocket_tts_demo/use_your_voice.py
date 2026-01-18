#!/usr/bin/env python3
"""
Use Your Own Voice Sample with Pocket TTS

This script shows how to use your own recorded audio file
to create a custom TTS voice.
"""

from pocket_tts import TTSModel
import scipy.io.wavfile
import sys
import os


def main():
    print("🎙️  Custom Voice Cloning Demo")
    print("="*60)

    # Check if user provided a voice file
    if len(sys.argv) < 2:
        print("\nUsage: python use_your_voice.py <your_voice_file.wav>")
        print("\nExample:")
        print("   python use_your_voice.py my_recording.wav")
        print("\nYou can also use a URL:")
        print("   python use_your_voice.py https://example.com/voice.wav")
        print("\nOr use a preset voice:")
        print("   python use_your_voice.py javert")
        print("\nAvailable presets:")
        print("   alba, marius, javert, jean, fantine, cosette, eponine, azelma")
        print("\n" + "="*60)
        print("\n📝 TIPS FOR RECORDING YOUR OWN VOICE:\n")
        print("1. Record 10-30 seconds of clear speech")
        print("2. Use WAV format (not MP3)")
        print("3. Speak with the characteristics you want (gritty, breathy, etc.)")
        print("4. Include varied intonation and complete sentences")
        print("5. Minimize background noise")
        print("\nExample text to record:")
        print('   "The old road winds through forgotten places,')
        print('    where shadows linger and secrets wait.')
        print('    Every step echoes with stories untold,')
        print('    carried on whispers of wind and dust."')
        print("\n" + "="*60)
        return

    voice_file = sys.argv[1]

    # Check if file exists (unless it's a URL or preset)
    if not voice_file.startswith("http") and voice_file not in ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]:
        if not os.path.exists(voice_file):
            print(f"\n❌ Error: File '{voice_file}' not found!")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   Make sure the file exists and the path is correct.\n")
            return

    print(f"\nUsing voice: {voice_file}")
    print("="*60 + "\n")

    # Load TTS model
    print("Loading Pocket TTS model...")
    tts_model = TTSModel.load_model()
    print(f"✓ Model loaded! Sample rate: {tts_model.sample_rate} Hz\n")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Load your custom voice
    print(f"Loading your custom voice from: {voice_file}")
    try:
        voice_state = tts_model.get_state_for_audio_prompt(voice_file)
        print("✓ Voice loaded successfully!\n")
    except Exception as e:
        print(f"\n❌ Error loading voice: {e}")
        print("\nTroubleshooting:")
        print("  • Make sure the file is a valid audio file (WAV, MP3, etc.)")
        print("  • Check that the file isn't corrupted")
        print("  • Try converting to WAV format with: ffmpeg -i input.mp3 output.wav")
        return

    # Generate speech with different texts
    print("Generating speech samples with your custom voice...")
    print("-" * 60)

    test_cases = [
        {
            "text": "Hello, this is a test of my custom voice. How does it sound?",
            "file": "output/custom_test1.wav",
            "desc": "Basic test"
        },
        {
            "text": "The wind howled through the canyon, a low whisper of secrets long forgotten.",
            "file": "output/custom_test2.wav",
            "desc": "Atmospheric narration"
        },
        {
            "text": "In the depths of the ancient forest, shadows dance among the twisted trees.",
            "file": "output/custom_test3.wav",
            "desc": "Dramatic scene"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['desc']}")
        print(f"   Text: '{test['text']}'")
        print(f"   Generating...")

        audio = tts_model.generate_audio(voice_state, test["text"])
        scipy.io.wavfile.write(test["file"], tts_model.sample_rate, audio.numpy())

        print(f"   ✓ Saved to: {test['file']}")

    # Generate a longer sample
    print(f"\n4. Longer sample")
    long_text = """
    The old lighthouse stood sentinel against the gathering storm.
    Waves crashed against the rocks below, sending spray high into the air.
    Inside, a single candle flickered in the darkness,
    casting dancing shadows on the weathered walls.
    Years had passed since anyone had climbed these stairs,
    yet the light still burned, a beacon in the night.
    """
    print(f"   Generating longer narrative...")

    audio = tts_model.generate_audio(voice_state, long_text.strip())
    long_file = "output/custom_long_sample.wav"
    scipy.io.wavfile.write(long_file, tts_model.sample_rate, audio.numpy())
    print(f"   ✓ Saved to: {long_file}")

    # Done!
    print("\n" + "="*60)
    print("🎉 Done! Your custom voice samples have been generated!")
    print("="*60)
    print("\n📁 Generated files:")
    for test in test_cases:
        print(f"   • {test['file']}")
    print(f"   • {long_file}")

    print("\n🎧 Next steps:")
    print("   1. Listen to the generated files")
    print("   2. If you don't like the results:")
    print("      - Try a different voice sample")
    print("      - Make sure your sample is 10-30 seconds")
    print("      - Ensure good audio quality")
    print("   3. Use the voice in your own projects!")

    print("\n💡 To generate more:")
    print(f"   python use_your_voice.py {voice_file}")
    print("   (Edit this script to add your own text)")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
