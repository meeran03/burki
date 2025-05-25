"""
Test the audio denoising service functionality.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch

from app.services.audio_denoising_service import AudioDenoisingService


class TestAudioDenoisingService:
    """Test cases for the AudioDenoisingService."""

    @pytest.fixture
    def denoising_service(self):
        """Create a denoising service instance for testing."""
        return AudioDenoisingService(call_sid="test_call_123", enabled=True)

    @pytest.mark.asyncio
    async def test_initialization(self, denoising_service):
        """Test that the denoising service initializes correctly."""
        success = await denoising_service.initialize()
        assert success is True
        assert denoising_service.is_initialized is True
        assert denoising_service.denoising_method in ["rnnoise", "fallback"]

    @pytest.mark.asyncio
    async def test_disabled_service(self):
        """Test that disabled service returns original audio."""
        service = AudioDenoisingService(call_sid="test_call_disabled", enabled=False)
        await service.initialize()
        
        # Create test audio data (μ-law encoded)
        test_audio = b'\x00\x01\x02\x03\x04\x05\x06\x07'
        
        result = await service.process_audio(test_audio)
        assert result == test_audio  # Should return original audio unchanged

    def test_mulaw_conversion(self, denoising_service):
        """Test μ-law to PCM conversion and back."""
        # Create test μ-law data
        mulaw_data = bytes([0x00, 0x7F, 0x80, 0xFF])
        
        # Convert to PCM
        pcm_data = denoising_service._convert_mulaw_to_pcm(mulaw_data)
        assert isinstance(pcm_data, np.ndarray)
        assert pcm_data.dtype == np.float32
        
        # Convert back to μ-law
        mulaw_result = denoising_service._convert_pcm_to_mulaw(pcm_data)
        assert isinstance(mulaw_result, bytes)
        assert len(mulaw_result) == len(mulaw_data)

    @pytest.mark.asyncio
    async def test_audio_processing_with_fallback(self):
        """Test audio processing with fallback filters."""
        service = AudioDenoisingService(call_sid="test_fallback", enabled=True)
        
        # Mock the RNNoise wrapper to force fallback mode
        with patch('app.services.audio_denoising_service.get_rnnoise_wrapper') as mock_wrapper:
            mock_wrapper.return_value.is_available = False
            
            await service.initialize()
            assert service.denoising_method == "fallback"
            
            # Create test audio data (160 samples of μ-law)
            # This represents 20ms of audio at 8kHz
            test_audio = bytes(range(160))
            
            result = await service.process_audio(test_audio)
            assert isinstance(result, bytes)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_frame_processing(self, denoising_service):
        """Test processing of individual audio frames."""
        await denoising_service.initialize()
        
        # Create a test frame (160 samples)
        test_frame = np.random.randn(160).astype(np.float32) * 0.1
        
        processed_frame = denoising_service._process_frame_with_rnnoise(test_frame)
        
        assert isinstance(processed_frame, np.ndarray)
        assert len(processed_frame) == len(test_frame)
        assert processed_frame.dtype == np.float32

    @pytest.mark.asyncio
    async def test_cleanup(self, denoising_service):
        """Test cleanup functionality."""
        await denoising_service.initialize()
        
        # Process some frames to generate statistics
        test_audio = bytes(range(160))
        await denoising_service.process_audio(test_audio)
        
        # Cleanup should not raise any exceptions
        await denoising_service.cleanup()

    def test_get_stats(self, denoising_service):
        """Test statistics collection."""
        stats = denoising_service.get_stats()
        
        assert isinstance(stats, dict)
        assert "enabled" in stats
        assert "initialized" in stats
        assert "method" in stats
        assert "frames_processed" in stats
        assert "call_sid" in stats
        assert stats["call_sid"] == "test_call_123"

    @pytest.mark.asyncio
    async def test_error_handling(self, denoising_service):
        """Test error handling in audio processing."""
        await denoising_service.initialize()
        
        # Test with invalid audio data
        invalid_audio = b''
        result = await denoising_service.process_audio(invalid_audio)
        assert result == invalid_audio  # Should return original on error

    @pytest.mark.asyncio
    async def test_buffer_accumulation(self, denoising_service):
        """Test that audio buffer accumulates correctly."""
        await denoising_service.initialize()
        
        # Send partial frame (less than 160 samples)
        partial_audio = bytes(range(80))  # Half a frame
        
        result1 = await denoising_service.process_audio(partial_audio)
        # Should return original since no complete frame is available
        assert result1 == partial_audio
        
        # Send another partial frame to complete a full frame
        result2 = await denoising_service.process_audio(partial_audio)
        # Now we should have processed audio
        assert isinstance(result2, bytes)


def test_simple_noise_gate():
    """Test the SimpleNoiseGate fallback filter."""
    from app.services.rnnoise_wrapper import SimpleNoiseGate
    
    gate = SimpleNoiseGate(threshold=0.1, attack_time=0.001, release_time=0.1)
    
    # Test with loud signal (above threshold)
    loud_frame = np.ones(160) * 0.5  # Loud signal
    result = gate.process_frame(loud_frame)
    assert np.mean(result) > 0  # Should pass through
    
    # Test with quiet signal (below threshold)
    quiet_frame = np.ones(160) * 0.01  # Quiet signal
    result = gate.process_frame(quiet_frame)
    # After processing, signal should be attenuated


def test_adaptive_filter():
    """Test the AdaptiveFilter fallback filter."""
    from app.services.rnnoise_wrapper import AdaptiveFilter
    
    filter = AdaptiveFilter(filter_length=16, step_size=0.01)
    
    # Test with a simple signal
    test_frame = np.sin(np.linspace(0, 2*np.pi, 160))
    result = filter.process_frame(test_frame)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == len(test_frame)
    assert result.dtype == np.float64  # Default numpy dtype


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        service = AudioDenoisingService(call_sid="simple_test", enabled=True)
        await service.initialize()
        
        print(f"Denoising method: {service.denoising_method}")
        print(f"Initialized: {service.is_initialized}")
        
        # Test with some audio
        test_audio = bytes(range(160))
        result = await service.process_audio(test_audio)
        
        print(f"Processed {len(test_audio)} bytes -> {len(result)} bytes")
        print("Stats:", service.get_stats())
        
        await service.cleanup()
        print("Test completed successfully!")

    asyncio.run(simple_test()) 