#!/usr/bin/env python3
"""
Convert audio recordings to the format needed for Pocket TTS.

Use this if you recorded audio on Windows or have an existing audio file
that needs to be converted to the right format (24kHz, mono, WAV).
"""

from pydub import AudioSegment
import sys
import os


def convert_to_tts_format(input_file, output_file="my_voice.wav"):
    """
    Convert audio file to Pocket TTS format.

    Args:
        input_file: Path to input audio file (WAV, MP3, etc.)
        output_file: Path for output WAV file (default: my_voice.wav)
    """
    print("🎛️  Audio Converter for Pocket TTS")
    print("="*60)

    # Check input file exists
    if not os.path.exists(input_file):
        print(f"\n❌ Error: File not found: {input_file}")
        print("\nIf you recorded on Windows, copy it to WSL first:")
        print("  cp /mnt/c/Users/YourName/Downloads/recording.wav .")
        sys.exit(1)

    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")

    try:
        # Load audio file
        print("\n📂 Loading audio file...")
        audio = AudioSegment.from_file(input_file)

        print(f"   Original: {audio.frame_rate}Hz, {audio.channels} channel(s), {len(audio)}ms")

        # Convert to Pocket TTS format
        print("\n🔧 Converting to Pocket TTS format...")

        # Convert to mono
        if audio.channels > 1:
            print("   • Converting to mono")
            audio = audio.set_channels(1)

        # Convert to 24kHz (Pocket TTS sample rate)
        if audio.frame_rate != 24000:
            print(f"   • Resampling to 24000 Hz (was {audio.frame_rate} Hz)")
            audio = audio.set_frame_rate(24000)

        # Normalize audio (make it use the full dynamic range)
        print("   • Normalizing volume")
        audio = audio.normalize()

        # Export as WAV
        print(f"\n💾 Saving as {output_file}...")
        audio.export(output_file, format="wav", parameters=["-q:a", "0"])

        duration_sec = len(audio) / 1000.0
        print(f"✓ Converted successfully!")
        print(f"   Duration: {duration_sec:.1f} seconds")
        print(f"   Format: 24000 Hz, mono, WAV")

        # Next steps
        print("\n" + "="*60)
        print("🎉 Done! Your audio is ready for voice cloning!")
        print("="*60)

        print("\n🎧 Review the converted file:")
        print(f"   ffplay {output_file}")

        print("\n🚀 Use it with Pocket TTS:")
        print(f"   python use_your_voice.py {output_file}")
        print("\n   OR directly with the CLI:")
        print(f"   uvx pocket-tts generate --voice {output_file} --text 'Your text here'")

        print("\n⚠️  NOTE: Voice cloning from custom WAV files requires")
        print("   Hugging Face authentication. See README.md Method 2.")

        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  • Make sure the input file is a valid audio file")
        print("  • Supported formats: WAV, MP3, M4A, OGG, FLAC, etc.")
        print("  • Check that pydub is installed: uv pip install pydub")
        print("  • Make sure ffmpeg is installed: sudo apt install ffmpeg")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("🎛️  Audio Converter for Pocket TTS")
        print("="*60)
        print("\nUsage: python convert_recording.py <input_file> [output_file]")
        print("\nExample:")
        print("   python convert_recording.py recording.wav")
        print("   python convert_recording.py /mnt/c/Users/Name/Downloads/voice.mp3 my_voice.wav")
        print("\nThis converts any audio file to Pocket TTS format:")
        print("   • 24000 Hz sample rate")
        print("   • Mono (single channel)")
        print("   • WAV format")
        print("   • Normalized volume")
        print("\n" + "="*60)
        print("\n💡 TIP: If you recorded on Windows:")
        print("   1. First copy to WSL: cp /mnt/c/Users/Name/Downloads/recording.wav .")
        print("   2. Then convert: python convert_recording.py recording.wav")
        print("="*60)
        sys.exit(0)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "my_voice.wav"

    convert_to_tts_format(input_file, output_file)


if __name__ == "__main__":
    main()
