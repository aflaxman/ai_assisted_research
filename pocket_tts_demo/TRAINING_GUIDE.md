# Training & Voice Customization Guide

**Note:** This guide has been merged into the main README.md for easier access.

## Quick Links

For complete instructions on voice customization, see the README.md:
- **Method 1**: Automated Voice Processing (create_custom_voice.py)
- **Method 2**: Record and Clone Your Own Voice (record_voice.py)
- **Method 3**: Training Information (why it's not available)

## Summary

- **Training code is NOT available** - Pocket TTS is released as a pre-trained model only
- **Voice cloning IS available** - Clone any voice from a 10-30 second audio sample
- **Authentication required** - Custom voice cloning needs Hugging Face authentication
- **Preset voices work without auth** - 8 voices available out of the box

See README.md for:
- Step-by-step recording instructions
- Authentication setup
- Tips for creating gritty/breathy voices
- Complete workflow from recording to generation

## Alternative TTS Models with Training Support

If you absolutely need to train a model:

1. **Coqui TTS** - Full training pipeline available
   ```bash
   pip install TTS
   ```

2. **XTTS** - Fine-tuning supported on custom datasets

3. **Piper TTS** - Lightweight with training documentation

For Pocket TTS, stick with voice cloning - it's simpler and gives great results!
