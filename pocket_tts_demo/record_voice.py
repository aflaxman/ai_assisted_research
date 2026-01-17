#!/usr/bin/env python3
"""
Record audio from your microphone for voice cloning.

This script records a sample of your voice that can be used with Pocket TTS
for voice cloning. Speak in the style you want (gritty, breathy, etc.)
"""

import sounddevice as sd
import scipy.io.wavfile
import numpy as np
import sys


def record_voice(duration=15, sample_rate=24000):
    """
    Record audio from the default microphone.

    Args:
        duration: Recording duration in seconds (default: 15)
        sample_rate: Sample rate in Hz (default: 24000, matches Pocket TTS)
    """
    print("🎙️  Voice Recording for Pocket TTS")
    print("="*60)
    print(f"\nRecording Settings:")
    print(f"  • Duration: {duration} seconds")
    print(f"  • Sample Rate: {sample_rate} Hz")
    print(f"  • Format: WAV (mono)")

    print("\n📝 Tips for a good recording:")
    print("  • Use a quiet room with minimal background noise")
    print("  • Speak naturally and clearly")
    print("  • For gritty voice: lower pitch, add vocal fry/raspiness")
    print("  • For breathy voice: emphasize air flow, softer tone")
    print("  • Vary your intonation - don't speak monotone")

    print("\n💡 Suggested text to read:")
    print('  "The old road winds through forgotten places,')
    print('   where shadows linger and secrets wait.')
    print('   Every step echoes with stories untold,')
    print('   carried on whispers of wind and dust."')

    input(f"\n▶️  Press ENTER to start recording for {duration} seconds...")

    print(f"\n🔴 RECORDING... Speak now!")
    print("   (Recording will automatically stop)")

    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,  # Mono
        dtype=np.float32
    )
    sd.wait()  # Wait until recording is finished

    print("✓ Recording complete!")

    return recording, sample_rate


def save_recording(recording, sample_rate, filename="my_voice.wav"):
    """Save the recording to a WAV file."""
    # Convert to int16 for WAV format
    recording_int16 = np.int16(recording * 32767)

    scipy.io.wavfile.write(filename, sample_rate, recording_int16)
    print(f"\n💾 Saved recording to: {filename}")

    return filename


def main():
    # Get duration from command line or use default
    duration = 15
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print(f"Invalid duration: {sys.argv[1]}")
            print("Usage: python record_voice.py [duration_in_seconds]")
            sys.exit(1)

    # Get output filename from command line or use default
    output_file = "my_voice.wav"
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    try:
        # Record audio
        recording, sample_rate = record_voice(duration=duration)

        # Save to file
        filename = save_recording(recording, sample_rate, output_file)

        # Next steps
        print("\n" + "="*60)
        print("🎉 Done! Your voice sample is ready for cloning!")
        print("="*60)

        print("\n🎧 Review your recording:")
        print(f"   mpv {filename}")

        print("\n📋 If you're not happy with it:")
        print(f"   python record_voice.py {duration}  # Record again")

        print("\n🚀 Use it with Pocket TTS:")
        print(f"   python use_your_voice.py {filename}")
        print("\n   OR directly with the CLI:")
        print(f"   uvx pocket-tts generate --voice {filename} --text 'Your text here'")

        print("\n⚠️  NOTE: Voice cloning from custom WAV files requires")
        print("   Hugging Face authentication. See README.md for details.")

        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  • Make sure you have a microphone connected")
        print("  • Check that sounddevice is installed: pip install sounddevice")
        print("  • On Linux/WSL, you may need: sudo apt-get install portaudio19-dev")
        print("  • Test your mic with: python -m sounddevice")
        sys.exit(1)


if __name__ == "__main__":
    main()
