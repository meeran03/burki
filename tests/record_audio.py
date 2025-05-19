"""
Script to record audio for testing the Deepgram transcription service.
"""

import os
import argparse
import wave
import sys
import time
import logging
import pyaudio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def record_audio(output_file, duration=5, sample_rate=8000, channels=1):
    """
    Record audio from the microphone and save it to a file.
    
    Args:
        output_file: Path to save the recording
        duration: Recording duration in seconds
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
    """
    # Configure PyAudio
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024
    )
    
    logger.info(f"Recording for {duration} seconds...")
    
    # Record audio
    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
        # Print progress
        sys.stdout.write(f"\rRecording: {i * 1024 / sample_rate:.1f}s / {duration}s")
        sys.stdout.flush()
    
    logger.info("\nRecording complete!")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save to WAV file
    logger.info(f"Saving to {output_file}")
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    logger.info(f"Saved {duration} seconds of audio to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record audio for testing")
    parser.add_argument("--output", default="test_audio.wav", help="Output file path")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=8000, help="Sample rate (8000 for Twilio compatibility)")
    args = parser.parse_args()
    
    # Create recordings directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    # Record audio
    record_audio(args.output, args.duration, args.sample_rate) 