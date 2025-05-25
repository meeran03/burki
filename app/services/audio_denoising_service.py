"""
Audio denoising service using RNNoise for real-time noise suppression.
This service processes audio in real-time without adding significant latency.
"""

import logging
import asyncio
import base64
import numpy as np
from typing import Optional, Callable, Dict, Any
import subprocess
import os
from pathlib import Path

# Import our RNNoise wrapper and fallback filters
from app.services.rnnoise_wrapper import (
    get_rnnoise_wrapper, 
    SimpleNoiseGate, 
    AdaptiveFilter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AudioDenoisingService:
    """
    Real-time audio denoising service using RNNoise with fallback filters.
    Processes audio streams with minimal latency impact.
    """

    def __init__(self, call_sid: Optional[str] = None, enabled: bool = True):
        """
        Initialize the audio denoising service.

        Args:
            call_sid: The unique identifier for this call
            enabled: Whether denoising is enabled for this call
        """
        self.call_sid = call_sid
        self.enabled = enabled
        self.is_initialized = False
        
        # Audio processing parameters
        self.sample_rate = 8000  # Twilio uses 8kHz
        self.frame_size = 160    # RNNoise expects 160 samples per frame (20ms at 8kHz)
        self.channels = 1        # Mono audio
        
        # Audio buffer for accumulating samples
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Denoising components
        self.rnnoise_wrapper = None
        self.noise_gate = None
        self.adaptive_filter = None
        self.denoising_method = "none"  # Will be set during initialization
        
        # Statistics
        self.frames_processed = 0
        self.total_processing_time = 0.0
        
        logger.info(f"AudioDenoisingService initialized for call {call_sid} (enabled: {enabled})")

    async def initialize(self) -> bool:
        """
        Initialize the denoising engine with the best available method.
        
        Returns:
            bool: Whether initialization was successful
        """
        if not self.enabled:
            logger.info(f"Denoising disabled for call {self.call_sid}")
            return True
            
        try:
            # Try to use RNNoise first
            self.rnnoise_wrapper = get_rnnoise_wrapper()
            
            if self.rnnoise_wrapper.is_available:
                self.denoising_method = "rnnoise"
                logger.info(f"Using RNNoise for call {self.call_sid}")
            else:
                # Fall back to simple noise gate + adaptive filter
                self.noise_gate = SimpleNoiseGate(
                    threshold=0.005,  # Lower threshold for better sensitivity
                    attack_time=0.001,  # Fast attack
                    release_time=0.05   # Moderate release
                )
                
                self.adaptive_filter = AdaptiveFilter(
                    filter_length=16,   # Shorter filter for lower latency
                    step_size=0.005     # Conservative learning rate
                )
                
                self.denoising_method = "fallback"
                logger.info(f"Using fallback denoising (noise gate + adaptive filter) for call {self.call_sid}")
            
            self.is_initialized = True
            logger.info(f"Audio denoising initialized successfully for call {self.call_sid} using {self.denoising_method}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audio denoising for call {self.call_sid}: {e}")
            self.enabled = False  # Disable denoising if initialization fails
            return False

    def _convert_mulaw_to_pcm(self, mulaw_data: bytes) -> np.ndarray:
        """
        Convert μ-law encoded audio to PCM float32.
        
        Args:
            mulaw_data: μ-law encoded audio bytes
            
        Returns:
            np.ndarray: PCM audio as float32 array
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
            
            # Normalize to [-1, 1] range
            pcm_float = linear.astype(np.float32) / 8159.0  # μ-law max value
            
            return pcm_float
            
        except Exception as e:
            logger.error(f"Error converting μ-law to PCM: {e}")
            return np.array([], dtype=np.float32)

    def _convert_pcm_to_mulaw(self, pcm_data: np.ndarray) -> bytes:
        """
        Convert PCM float32 audio to μ-law encoded bytes.
        
        Args:
            pcm_data: PCM audio as float32 array
            
        Returns:
            bytes: μ-law encoded audio bytes
        """
        try:
            # Clamp to [-1, 1] range and scale
            pcm_clamped = np.clip(pcm_data, -1.0, 1.0)
            pcm_scaled = (pcm_clamped * 8159.0).astype(np.int16)
            
            # Convert to μ-law (ITU-T G.711)
            abs_pcm = np.abs(pcm_scaled)
            sign = (pcm_scaled < 0).astype(np.uint8)
            
            # Add bias
            abs_pcm = abs_pcm + 33
            
            # Find exponent
            exponent = np.zeros_like(abs_pcm, dtype=np.uint8)
            for i in range(7, -1, -1):
                mask = abs_pcm >= (1 << (i + 5))
                exponent = np.where(mask & (exponent == 0), i, exponent)
            
            # Calculate mantissa
            mantissa = (abs_pcm >> (exponent + 1)) & 0x0F
            
            # Combine components
            mulaw = (sign << 7) | (exponent << 4) | mantissa
            
            # Invert bits (μ-law is stored inverted)
            mulaw = mulaw ^ 0xFF
            
            return mulaw.astype(np.uint8).tobytes()
            
        except Exception as e:
            logger.error(f"Error converting PCM to μ-law: {e}")
            return b''

    def _process_frame_with_rnnoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single audio frame with RNNoise.
        
        Args:
            frame: Audio frame as float32 array (160 samples)
            
        Returns:
            np.ndarray: Denoised audio frame
        """
        if not self.enabled or not self.is_initialized:
            return frame
            
        try:
            if self.denoising_method == "rnnoise" and self.rnnoise_wrapper:
                # Convert to int16 for RNNoise
                frame_int16 = (frame * 32767).astype(np.int16)
                frame_bytes = frame_int16.tobytes()
                
                # Process with RNNoise (this is a simplified interface)
                # In a real implementation, you'd use the actual RNNoise C API
                processed_bytes = self.rnnoise_wrapper.process_audio_file(frame_bytes)
                
                # Convert back to float32
                if len(processed_bytes) > 0:
                    processed_int16 = np.frombuffer(processed_bytes, dtype=np.int16)
                    processed_float = processed_int16.astype(np.float32) / 32767.0
                    return processed_float[:len(frame)]  # Ensure same length
                else:
                    return frame
                    
            elif self.denoising_method == "fallback":
                # Use fallback filters
                processed_frame = frame
                
                # Apply noise gate first
                if self.noise_gate:
                    processed_frame = self.noise_gate.process_frame(processed_frame)
                
                # Apply adaptive filter
                if self.adaptive_filter:
                    processed_frame = self.adaptive_filter.process_frame(processed_frame)
                
                return processed_frame
            else:
                return frame
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame  # Return original frame on error

    async def process_audio(self, audio_data: bytes) -> bytes:
        """
        Process incoming audio data with denoising.
        
        Args:
            audio_data: Raw μ-law encoded audio bytes from Twilio
            
        Returns:
            bytes: Processed μ-law encoded audio bytes
        """
        if not self.enabled or not self.is_initialized:
            return audio_data
            
        try:
            # For now, just pass through the audio without processing
            # The μ-law conversion and frame processing was causing issues
            # TODO: Implement proper real-time audio denoising
            logger.debug(f"Audio denoising pass-through for call {self.call_sid}: {len(audio_data)} bytes")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error processing audio for call {self.call_sid}: {e}")
            return audio_data  # Return original audio on error

    async def cleanup(self) -> None:
        """
        Clean up resources when the call ends.
        """
        try:
            # Log final statistics
            if self.frames_processed > 0:
                avg_time = self.total_processing_time / self.frames_processed * 1000
                logger.info(f"Denoising session ended for {self.call_sid} ({self.denoising_method}): "
                          f"{self.frames_processed} frames processed, "
                          f"{avg_time:.2f}ms avg processing time per frame")
            
            logger.info(f"AudioDenoisingService cleaned up for call {self.call_sid}")
            
        except Exception as e:
            logger.error(f"Error cleaning up denoising service for call {self.call_sid}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get denoising performance statistics.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        avg_time = 0.0
        if self.frames_processed > 0:
            avg_time = self.total_processing_time / self.frames_processed * 1000
            
        return {
            "enabled": self.enabled,
            "initialized": self.is_initialized,
            "method": self.denoising_method,
            "frames_processed": self.frames_processed,
            "total_processing_time_ms": self.total_processing_time * 1000,
            "avg_processing_time_ms": avg_time,
            "call_sid": self.call_sid,
        }


# Utility functions for RNNoise integration

def check_rnnoise_availability() -> bool:
    """
    Check if RNNoise WASM module is available.
    
    Returns:
        bool: Whether RNNoise is available
    """
    try:
        wrapper = get_rnnoise_wrapper()
        return wrapper.is_available
    except Exception:
        return False


async def compile_rnnoise_wasm() -> bool:
    """
    Compile RNNoise to WebAssembly if not already available.
    This would be run during application startup.
    
    Returns:
        bool: Whether compilation was successful
    """
    try:
        if check_rnnoise_availability():
            logger.info("RNNoise already available")
            return True
            
        logger.info("RNNoise not available, using fallback denoising methods")
        return True  # Fallback methods are always available
        
    except Exception as e:
        logger.error(f"Error checking RNNoise: {e}")
        return True  # Fallback methods are always available 