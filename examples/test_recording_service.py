#!/usr/bin/env python3
"""
Test script for the RecordingService functionality.
This demonstrates how to use the local audio recording capabilities.
"""

import asyncio
import os
import sys
import time
import wave
import random

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.recording_service import RecordingService


async def test_recording_service():
    """Test the RecordingService with simulated audio data."""
    
    print("Testing RecordingService...")
    
    # Test both WAV and MP3 formats
    for format_type in ["wav", "mp3"]:
        print(f"\n🎵 Testing {format_type.upper()} format...")
        
        # Create a test call SID
        call_sid = f"test_call_{format_type}_{int(time.time())}"
        
        # Initialize recording service
        recording_service = RecordingService(
            call_sid=call_sid,
            enabled=True,
            format=format_type,
            sample_rate=8000,
            channels=1,
            record_user=True,
            record_assistant=True,
            record_mixed=True,
            auto_save=True,
            recordings_dir="test_recordings",
        )
        
        # Set up callbacks
        async def recording_started_callback(call_sid: str):
            print(f"✅ Recording started for call {call_sid}")
            
        async def recording_stopped_callback(call_sid: str):
            print(f"⏹️  Recording stopped for call {call_sid}")
            
        async def recording_saved_callback(call_sid: str, saved_files: dict):
            print(f"💾 Recordings saved for call {call_sid}:")
            for recording_type, file_path in saved_files.items():
                print(f"   - {recording_type}: {file_path}")
        
        recording_service.set_callbacks(
            recording_started_callback=recording_started_callback,
            recording_stopped_callback=recording_stopped_callback,
            recording_saved_callback=recording_saved_callback,
        )
        
        # Start recording
        print(f"\n🎙️  Starting {format_type.upper()} recording...")
        success = await recording_service.start_recording()
        if not success:
            print(f"❌ Failed to start {format_type} recording")
            continue
        
        # Simulate some audio data
        print(f"🎵 Simulating audio data for {format_type.upper()}...")
        
        # Generate some fake μ-law encoded audio data (simulating Twilio/ElevenLabs format)
        sample_rate = 8000
        duration_seconds = 3  # Shorter duration for testing
        samples_per_chunk = 160  # 20ms chunks at 8kHz
        
        def generate_mulaw_chunk(chunk_size: int, amplitude: float = 0.1) -> bytes:
            """Generate a chunk of μ-law encoded audio data."""
            # Generate PCM samples (sine wave with some noise)
            import math
            pcm_samples = []
            for i in range(chunk_size):
                # Generate a sine wave with some random noise
                t = i / sample_rate
                sine_wave = math.sin(2 * math.pi * 440 * t) * amplitude  # 440 Hz tone
                noise = (random.random() - 0.5) * 0.05  # Small amount of noise
                sample = sine_wave + noise
                
                # Clamp to [-1, 1] and convert to 16-bit
                sample = max(-1.0, min(1.0, sample))
                pcm_16bit = int(sample * 32767)
                pcm_samples.append(pcm_16bit)
            
            # Convert PCM to μ-law (simplified conversion)
            mulaw_bytes = bytearray()
            for pcm_sample in pcm_samples:
                # Simple μ-law encoding (not perfect but good enough for testing)
                abs_sample = abs(pcm_sample)
                if abs_sample < 33:
                    mulaw_val = 0
                else:
                    # Find the exponent
                    exponent = 7
                    for exp in range(7):
                        if abs_sample < (33 << exp):
                            exponent = exp
                            break
                    
                    # Calculate mantissa
                    mantissa = (abs_sample >> (exponent + 1)) & 0x0F
                    
                    # Combine sign, exponent, and mantissa
                    sign = 0x80 if pcm_sample < 0 else 0x00
                    mulaw_val = sign | (exponent << 4) | mantissa
                
                # Invert bits (μ-law is stored inverted)
                mulaw_val = mulaw_val ^ 0xFF
                mulaw_bytes.append(mulaw_val)
            
            return bytes(mulaw_bytes)
        
        for i in range(int(duration_seconds * sample_rate / samples_per_chunk)):
            # Generate fake μ-law audio data
            audio_data = generate_mulaw_chunk(samples_per_chunk)
            
            # Alternate between user and assistant audio
            if i % 2 == 0:
                await recording_service.record_user_audio(audio_data)
            else:
                await recording_service.record_assistant_audio(audio_data)
            
            # Small delay to simulate real-time audio
            await asyncio.sleep(0.02)  # 20ms
            
            # Print progress
            if i % 50 == 0:  # Every second
                progress = (i * samples_per_chunk) / sample_rate
                print(f"   Recording progress: {progress:.1f}s / {duration_seconds}s")
        
        print(f"✅ Audio simulation complete for {format_type.upper()}")
        
        # Stop recording
        print(f"\n⏹️  Stopping {format_type.upper()} recording...")
        success = await recording_service.stop_recording()
        if not success:
            print(f"❌ Failed to stop {format_type} recording")
            continue
        
        # Get recording info
        print(f"\n📊 Recording information for {format_type.upper()}:")
        info = recording_service.get_recording_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Clean up
        print(f"\n🧹 Cleaning up {format_type.upper()} recording service...")
        await recording_service.cleanup()
        
        print(f"\n✅ {format_type.upper()} test completed successfully!")
        print(f"Check the 'test_recordings/{call_sid}' directory for the recorded files.")
    
    print(f"\n🎉 All format tests completed!")


async def test_recording_info():
    """Test getting recording information."""
    
    print("\n" + "="*50)
    print("Testing recording info functionality...")
    
    # Create a disabled recording service
    recording_service = RecordingService(
        call_sid="test_info_call",
        enabled=False,
    )
    
    info = recording_service.get_recording_info()
    print("\nRecording info for disabled service:")
    for key, value in info.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    print("🎙️  RecordingService Test Suite")
    print("="*50)
    
    # Run the tests
    asyncio.run(test_recording_service())
    asyncio.run(test_recording_info())
    
    print("\n🎉 All tests completed!") 