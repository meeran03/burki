# Local Audio Recording Setup

This document explains how to configure and use the local audio recording functionality in Burqi.

## Overview

Burqi now supports local audio recording during calls, which allows you to record audio streams directly on your server. This is in addition to the existing Twilio call recording functionality.

### Features

- **Separate Audio Streams**: Record user audio, assistant audio, or both
- **Mixed Audio Recording**: Create a combined recording with both user and assistant audio
- **Configurable Formats**: Support for WAV format (MP3 support can be added)
- **Assistant-Level Configuration**: Each assistant can have different recording settings
- **Database Integration**: Automatically create database records for recordings
- **Callback Support**: Get notified when recordings start, stop, or are saved

## Audio Format Handling

### Input Audio Formats

The recording service handles different audio formats from various sources:

- **User Audio (Twilio)**: μ-law encoded at 8kHz mono
- **Assistant Audio (ElevenLabs)**: μ-law encoded at 8kHz mono (configured as `ulaw_8000`)

### Format Conversion

All incoming audio is automatically converted from μ-law to high-quality PCM using Python's built-in `audioop` module:

1. **μ-law Decoding**: Raw μ-law bytes are decoded using `audioop.ulaw2lin()` for professional-quality conversion
2. **Quality Enhancement**: For MP3 format, audio is upsampled from 8kHz to 22.05kHz to match Twilio's quality
3. **High-Quality Export**: MP3 files are exported at 320kbps with professional encoding settings

This ensures that recorded files are:
- **Professional Quality**: Using industry-standard μ-law decoding
- **High Fidelity**: MP3 recordings at 22.05kHz match Twilio's quality
- **Optimized Encoding**: 320kbps MP3 with highest quality settings
- **Compatible**: Standard format playable in all audio players

### Technical Details

- **Input Format**: μ-law (8-bit compressed)
- **Decoding Method**: Python `audioop.ulaw2lin()` (professional quality)
- **WAV Output**: 16-bit PCM at 8kHz (original sample rate)
- **MP3 Output**: 320kbps, 22.05kHz (upsampled for quality)
- **File Size**: 
  - WAV: ~16 KB per second
  - MP3: ~40 KB per second (high quality)

## Configuration

### Assistant-Level Settings

Recording settings are configured per assistant in the `recording_settings` JSON field:

```json
{
  "enabled": true,
  "format": "mp3",
  "sample_rate": 8000,
  "channels": 1,
  "record_user_audio": true,
  "record_assistant_audio": true,
  "record_mixed_audio": true,
  "auto_save": true,
  "recordings_dir": "recordings",
  "create_database_records": true
}
```

### Configuration Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enabled` | boolean | `false` | Whether local recording is enabled |
| `format` | string | `"mp3"` | Audio format ("wav", "mp3") - **MP3 recommended for better quality** |
| `sample_rate` | integer | `8000` | Audio sample rate in Hz (upsampled to 22050 for MP3) |
| `channels` | integer | `1` | Number of audio channels (1 = mono, 2 = stereo) |
| `record_user_audio` | boolean | `true` | Whether to record user/caller audio |
| `record_assistant_audio` | boolean | `true` | Whether to record assistant/AI audio |
| `record_mixed_audio` | boolean | `true` | Whether to record mixed audio (both streams) |
| `auto_save` | boolean | `true` | Whether to automatically save recordings when call ends |
| `recordings_dir` | string | `"recordings"` | Directory to save recordings |
| `create_database_records` | boolean | `true` | Whether to create database records for recordings |

## Database Schema

The recording functionality uses the existing `recordings` table with these relevant fields:

- `recording_source`: Set to `"local"` for local recordings
- `recording_type`: Set to `"user"`, `"assistant"`, or `"mixed"`
- `file_path`: Local file path to the recording
- `format`: Audio format (e.g., "wav")
- `status`: Recording status ("recording", "completed", "failed")

## File Organization

Recordings are organized in the following directory structure:

```
recordings/
├── {call_sid}/
│   ├── user_{timestamp}.wav
│   ├── assistant_{timestamp}.wav
│   └── mixed_{timestamp}.wav
```

Where:
- `{call_sid}` is the Twilio call SID
- `{timestamp}` is the Unix timestamp when recording started

## Usage Examples

### Enabling Recording for an Assistant

```python
# Update assistant recording settings
assistant.recording_settings = {
    "enabled": True,
    "format": "mp3",
    "sample_rate": 8000,
    "channels": 1,
    "record_user_audio": True,
    "record_assistant_audio": True,
    "record_mixed_audio": True,
    "auto_save": True,
    "recordings_dir": "recordings",
    "create_database_records": True
}
```

### Programmatic Usage

```python
from app.services.recording_service import RecordingService

# Create recording service
recording_service = RecordingService(
    call_sid="CA1234567890abcdef",
    enabled=True,
    format="mp3",
    sample_rate=8000,
    channels=1,
    record_user=True,
    record_assistant=True,
    record_mixed=True,
    auto_save=True,
    recordings_dir="recordings",
)

# Set up callbacks
async def recording_started(call_sid: str):
    print(f"Recording started for {call_sid}")

async def recording_saved(call_sid: str, saved_files: dict):
    print(f"Recordings saved: {saved_files}")

recording_service.set_callbacks(
    recording_started_callback=recording_started,
    recording_saved_callback=recording_saved,
)

# Start recording
await recording_service.start_recording()

# Record audio data
await recording_service.record_user_audio(user_audio_bytes)
await recording_service.record_assistant_audio(assistant_audio_bytes)

# Stop recording
await recording_service.stop_recording()

# Clean up
await recording_service.cleanup()
```

## Integration with Call Handler

The recording service is automatically integrated with the call handler when enabled:

1. **Initialization**: Recording service is created during call setup
2. **User Audio**: Recorded in `handle_audio()` method after audio denoising
3. **Assistant Audio**: Recorded in `_handle_tts_audio()` method when sending TTS audio
4. **Cleanup**: Recording service is cleaned up when call ends

## Testing

You can test the recording functionality using the provided test script:

```bash
python examples/test_recording_service.py
```

This will:
- Create a test recording service
- Simulate audio data for 5 seconds
- Save recordings to `test_recordings/` directory
- Display recording information

## Storage Considerations

### Disk Space

Audio recordings can consume significant disk space:

- **8kHz mono WAV**: ~16 KB per second (~960 KB per minute)
- **16kHz mono WAV**: ~32 KB per second (~1.9 MB per minute)
- **44.1kHz stereo WAV**: ~176 KB per second (~10.6 MB per minute)

### Cleanup

Consider implementing automatic cleanup of old recordings:

```python
import os
import time
from pathlib import Path

def cleanup_old_recordings(recordings_dir: str, max_age_days: int = 30):
    """Remove recordings older than max_age_days."""
    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    
    for recording_file in Path(recordings_dir).rglob("*.wav"):
        if recording_file.stat().st_mtime < cutoff_time:
            recording_file.unlink()
            print(f"Deleted old recording: {recording_file}")
```

### Fixing Existing Garbled Recordings

If you have existing recordings that are garbled (recorded before the μ-law conversion fix), you can use the provided script to fix them:

```bash
# Fix a single recording
python scripts/fix_garbled_recordings.py path/to/garbled_recording.wav

# Fix all recordings in a directory
python scripts/fix_garbled_recordings.py recordings/ --recursive

# Fix with backup of original files
python scripts/fix_garbled_recordings.py recordings/ --recursive --backup

# Fix to a different output directory
python scripts/fix_garbled_recordings.py recordings/ --output fixed_recordings/ --recursive
```

The script will:
1. Read the garbled WAV file (which contains raw μ-law data)
2. Convert the μ-law data to proper 16-bit PCM
3. Save the fixed recording as a proper WAV file
4. Optionally create backups of original files

## Security Considerations

1. **File Permissions**: Ensure recording files have appropriate permissions
2. **Access Control**: Implement proper access controls for recording files
3. **Encryption**: Consider encrypting recordings at rest for sensitive data
4. **Retention Policies**: Implement data retention policies as required by regulations

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure the application has write permissions to the recordings directory
2. **Disk Space**: Monitor available disk space to prevent recording failures
3. **File Corruption**: Ensure proper cleanup of wave writers when calls end unexpectedly
4. **Garbled Audio**: This was caused by writing μ-law encoded audio directly to WAV files configured for PCM. The issue is now resolved with automatic μ-law to PCM conversion.

### Audio Quality Issues

If you experience audio quality problems:

1. **Check Input Format**: Ensure audio sources are providing μ-law encoded data
2. **Verify Conversion**: The service automatically converts μ-law to 16-bit PCM
3. **Sample Rate**: Verify all components use 8kHz sample rate
4. **File Playback**: Test recordings with standard audio players (VLC, QuickTime, etc.)

### Logging

Recording events are logged at various levels:

- `INFO`: Recording start/stop events, file saves
- `DEBUG`: Audio chunk processing
- `ERROR`: Recording failures, file I/O errors

### Monitoring

Monitor these metrics:
- Recording success/failure rates
- Disk space usage in recordings directory
- Recording file sizes and durations
- Database record creation success rates

## Future Enhancements

Potential improvements to the recording system:

1. **MP3 Support**: Add MP3 encoding for smaller file sizes
2. **Cloud Storage**: Upload recordings to S3/GCS automatically
3. **Real-time Streaming**: Stream recordings to external services
4. **Audio Processing**: Add noise reduction, volume normalization
5. **Compression**: Implement audio compression for storage efficiency
6. **Transcription Integration**: Automatically transcribe recordings 