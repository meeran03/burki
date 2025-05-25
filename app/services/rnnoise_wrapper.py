"""
Python wrapper for RNNoise audio denoising.
This provides a simple interface to RNNoise functionality.
"""

import logging
import numpy as np
import subprocess
import tempfile
import os
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class RNNoiseWrapper:
    """
    Python wrapper for RNNoise audio denoising.
    Provides real-time noise suppression capabilities.
    """

    def __init__(self):
        """Initialize the RNNoise wrapper."""
        self.is_available = False
        self.rnnoise_binary = None
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if RNNoise is available on the system."""
        try:
            # Try to find rnnoise_demo binary
            result = subprocess.run(
                ["which", "rnnoise_demo"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                self.rnnoise_binary = result.stdout.strip()
                self.is_available = True
                logger.info(f"RNNoise found at: {self.rnnoise_binary}")
            else:
                # Try alternative locations
                possible_paths = [
                    "/usr/local/bin/rnnoise_demo",
                    "/usr/bin/rnnoise_demo",
                    "./build/rnnoise/rnnoise/examples/rnnoise_demo",
                ]
                
                for path in possible_paths:
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        self.rnnoise_binary = path
                        self.is_available = True
                        logger.info(f"RNNoise found at: {self.rnnoise_binary}")
                        break
                
                if not self.is_available:
                    logger.warning("RNNoise binary not found. Audio denoising will be disabled.")
                    
        except Exception as e:
            logger.error(f"Error checking RNNoise availability: {e}")
            self.is_available = False

    def process_audio_file(self, input_data: bytes, sample_rate: int = 48000) -> bytes:
        """
        Process audio data through RNNoise.
        
        Args:
            input_data: Raw audio data (PCM 16-bit)
            sample_rate: Sample rate of the audio
            
        Returns:
            bytes: Processed audio data
        """
        if not self.is_available:
            return input_data
            
        try:
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as input_file:
                with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as output_file:
                    # Write input data
                    input_file.write(input_data)
                    input_file.flush()
                    
                    # Run RNNoise
                    cmd = [self.rnnoise_binary, input_file.name, output_file.name]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        timeout=10  # 10 second timeout
                    )
                    
                    if result.returncode == 0:
                        # Read processed data
                        with open(output_file.name, 'rb') as f:
                            processed_data = f.read()
                        
                        # Clean up temp files
                        os.unlink(input_file.name)
                        os.unlink(output_file.name)
                        
                        return processed_data
                    else:
                        logger.error(f"RNNoise processing failed: {result.stderr}")
                        # Clean up temp files
                        os.unlink(input_file.name)
                        os.unlink(output_file.name)
                        return input_data
                        
        except Exception as e:
            logger.error(f"Error processing audio with RNNoise: {e}")
            return input_data


class SimpleNoiseGate:
    """
    Simple noise gate implementation as a fallback when RNNoise is not available.
    """

    def __init__(self, threshold: float = 0.01, attack_time: float = 0.001, release_time: float = 0.1):
        """
        Initialize the noise gate.
        
        Args:
            threshold: Amplitude threshold below which audio is considered noise
            attack_time: Time to open the gate (in seconds)
            release_time: Time to close the gate (in seconds)
        """
        self.threshold = threshold
        self.attack_time = attack_time
        self.release_time = release_time
        self.gate_state = 0.0  # 0.0 = closed, 1.0 = open
        self.sample_rate = 8000

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process an audio frame through the noise gate.
        
        Args:
            frame: Audio frame as numpy array
            
        Returns:
            np.ndarray: Processed audio frame
        """
        if len(frame) == 0:
            return frame
            
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(frame ** 2))
        
        # Determine target gate state
        target_state = 1.0 if rms > self.threshold else 0.0
        
        # Calculate gate transition speed
        frame_time = len(frame) / self.sample_rate
        if target_state > self.gate_state:
            # Opening gate (attack)
            transition_speed = frame_time / self.attack_time
        else:
            # Closing gate (release)
            transition_speed = frame_time / self.release_time
        
        # Update gate state
        if target_state > self.gate_state:
            self.gate_state = min(1.0, self.gate_state + transition_speed)
        else:
            self.gate_state = max(0.0, self.gate_state - transition_speed)
        
        # Apply gate
        return frame * self.gate_state


class AdaptiveFilter:
    """
    Simple adaptive filter for noise reduction.
    """

    def __init__(self, filter_length: int = 32, step_size: float = 0.01):
        """
        Initialize the adaptive filter.
        
        Args:
            filter_length: Length of the adaptive filter
            step_size: Learning rate for the filter
        """
        self.filter_length = filter_length
        self.step_size = step_size
        self.weights = np.zeros(filter_length)
        self.input_buffer = np.zeros(filter_length)

    def process_sample(self, input_sample: float, reference_noise: float = 0.0) -> float:
        """
        Process a single audio sample.
        
        Args:
            input_sample: Input audio sample
            reference_noise: Reference noise sample (if available)
            
        Returns:
            float: Filtered audio sample
        """
        # Shift input buffer
        self.input_buffer[1:] = self.input_buffer[:-1]
        self.input_buffer[0] = input_sample
        
        # Calculate filter output
        filter_output = np.dot(self.weights, self.input_buffer)
        
        # Calculate error (desired signal - filter output)
        error = input_sample - filter_output
        
        # Update filter weights (LMS algorithm)
        self.weights += self.step_size * error * self.input_buffer
        
        return error

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process an audio frame.
        
        Args:
            frame: Input audio frame
            
        Returns:
            np.ndarray: Filtered audio frame
        """
        output = np.zeros_like(frame)
        for i, sample in enumerate(frame):
            output[i] = self.process_sample(sample)
        return output


def install_rnnoise() -> bool:
    """
    Install RNNoise from source.
    
    Returns:
        bool: Whether installation was successful
    """
    try:
        logger.info("Installing RNNoise from source...")
        
        # Create build directory
        build_dir = Path("build/rnnoise_install")
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone and build RNNoise
        commands = [
            ["git", "clone", "https://github.com/xiph/rnnoise.git"],
            ["./autogen.sh"],
            ["./configure", "--prefix=/usr/local"],
            ["make"],
            ["sudo", "make", "install"]
        ]
        
        os.chdir(build_dir)
        
        if not (build_dir / "rnnoise").exists():
            subprocess.run(commands[0], check=True)
        
        os.chdir("rnnoise")
        
        for cmd in commands[1:]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Error: {result.stderr}")
                return False
        
        logger.info("RNNoise installed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error installing RNNoise: {e}")
        return False


# Global instance for reuse
_rnnoise_wrapper = None

def get_rnnoise_wrapper() -> RNNoiseWrapper:
    """
    Get a global RNNoise wrapper instance.
    
    Returns:
        RNNoiseWrapper: The global wrapper instance
    """
    global _rnnoise_wrapper
    if _rnnoise_wrapper is None:
        _rnnoise_wrapper = RNNoiseWrapper()
    return _rnnoise_wrapper 