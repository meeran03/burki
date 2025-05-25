#!/usr/bin/env python3
"""
Script to fix garbled recordings that were saved as raw μ-law data in WAV files.
This script converts existing garbled recordings to proper PCM WAV format.
"""

import os
import sys
import wave
import argparse
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def convert_mulaw_to_pcm16(mulaw_data: bytes) -> bytes:
    """
    Convert μ-law encoded audio to 16-bit PCM.
    
    Args:
        mulaw_data: μ-law encoded audio bytes
        
    Returns:
        bytes: 16-bit PCM audio bytes suitable for WAV files
    """
    try:
        # Convert μ-law bytes to numpy array
        mulaw_array = np.frombuffer(mulaw_data, dtype=np.uint8)
        
        # μ-law to linear PCM conversion (ITU-T G.711)
        # Invert bits (μ-law is stored inverted)
        mulaw_array = mulaw_array ^ 0xFF
        
        # Extract sign, exponent, and mantissa
        sign = (mulaw_array & 0x80) >> 7
        exponent = (mulaw_array & 0x70) >> 4
        mantissa = mulaw_array & 0x0F
        
        # Convert to linear PCM
        linear = (mantissa << 1) + 33
        linear = linear << exponent
        linear = linear - 33
        
        # Apply sign
        linear = np.where(sign == 1, -linear, linear)
        
        # Scale to 16-bit range and convert to int16
        pcm16 = (linear * 4).astype(np.int16)  # Scale factor for proper volume
        
        return pcm16.tobytes()
        
    except Exception as e:
        print(f"Error converting μ-law to PCM16: {e}")
        return b''

def fix_garbled_recording(input_file: str, output_file: str = None) -> bool:
    """
    Fix a garbled recording by converting raw μ-law data to proper PCM WAV.
    
    Args:
        input_file: Path to the garbled WAV file
        output_file: Path for the fixed WAV file (defaults to input_file with _fixed suffix)
        
    Returns:
        bool: True if conversion was successful
    """
    if not output_file:
        # Create output filename with _fixed suffix
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")
    
    try:
        print(f"Processing: {input_file}")
        
        # Read the garbled WAV file
        with wave.open(input_file, 'rb') as wav_file:
            # Get WAV parameters
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            frames = wav_file.getnframes()
            
            print(f"  Original format: {channels} channels, {sample_width} bytes/sample, {framerate} Hz, {frames} frames")
            
            # Read all audio data
            raw_data = wav_file.readframes(frames)
        
        # If the file was recorded as raw μ-law in a WAV container,
        # the raw_data will be the μ-law bytes
        print(f"  Raw data size: {len(raw_data)} bytes")
        
        # Convert μ-law to PCM
        pcm_data = convert_mulaw_to_pcm16(raw_data)
        
        if not pcm_data:
            print(f"  Failed to convert μ-law data")
            return False
        
        print(f"  Converted to PCM: {len(pcm_data)} bytes")
        
        # Write the fixed WAV file
        with wave.open(output_file, 'wb') as fixed_wav:
            fixed_wav.setnchannels(1)  # Mono
            fixed_wav.setsampwidth(2)  # 16-bit
            fixed_wav.setframerate(8000)  # 8kHz
            fixed_wav.writeframes(pcm_data)
        
        print(f"  Fixed recording saved to: {output_file}")
        
        # Verify the output file
        with wave.open(output_file, 'rb') as verify_wav:
            verify_channels = verify_wav.getnchannels()
            verify_width = verify_wav.getsampwidth()
            verify_rate = verify_wav.getframerate()
            verify_frames = verify_wav.getnframes()
            
            print(f"  Fixed format: {verify_channels} channels, {verify_width} bytes/sample, {verify_rate} Hz, {verify_frames} frames")
        
        return True
        
    except Exception as e:
        print(f"Error fixing recording {input_file}: {e}")
        return False

def main():
    """Main function to process command line arguments and fix recordings."""
    parser = argparse.ArgumentParser(description="Fix garbled recordings by converting μ-law to PCM")
    parser.add_argument("input", help="Input WAV file or directory containing WAV files")
    parser.add_argument("--output", "-o", help="Output file or directory (optional)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    parser.add_argument("--backup", "-b", action="store_true", help="Create backup of original files")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        if args.backup:
            backup_path = str(input_path.parent / f"{input_path.stem}_backup{input_path.suffix}")
            print(f"Creating backup: {backup_path}")
            import shutil
            shutil.copy2(str(input_path), backup_path)
        
        success = fix_garbled_recording(str(input_path), args.output)
        if success:
            print("✅ Recording fixed successfully!")
        else:
            print("❌ Failed to fix recording")
            return 1
            
    elif input_path.is_dir():
        # Process directory
        pattern = "**/*.wav" if args.recursive else "*.wav"
        wav_files = list(input_path.glob(pattern))
        
        if not wav_files:
            print(f"No WAV files found in {input_path}")
            return 1
        
        print(f"Found {len(wav_files)} WAV files to process")
        
        success_count = 0
        for wav_file in wav_files:
            if args.backup:
                backup_path = str(wav_file.parent / f"{wav_file.stem}_backup{wav_file.suffix}")
                print(f"Creating backup: {backup_path}")
                import shutil
                shutil.copy2(str(wav_file), backup_path)
            
            output_file = None
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = str(output_dir / f"{wav_file.stem}_fixed{wav_file.suffix}")
            
            if fix_garbled_recording(str(wav_file), output_file):
                success_count += 1
        
        print(f"\n✅ Successfully fixed {success_count}/{len(wav_files)} recordings")
        
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 