"""
Example configuration for enabling audio denoising in Burqi assistants.

This file shows how to configure audio denoising for different use cases.
"""

# Example 1: Basic denoising configuration
basic_audio_settings = {
    "denoising_enabled": True,
}

# Example 2: Advanced denoising configuration
advanced_audio_settings = {
    "denoising_enabled": True,
    "noise_gate_threshold": 0.005,  # Lower = more sensitive to quiet sounds
    "adaptive_filter_length": 16,   # Shorter = lower latency, longer = better filtering
    "force_fallback": False,        # Set to True to skip RNNoise and use fallback filters
}

# Example 3: High-volume server configuration (optimized for performance)
high_volume_audio_settings = {
    "denoising_enabled": True,
    "force_fallback": True,         # Use lightweight fallback filters
    "noise_gate_threshold": 0.01,   # Less aggressive noise gating
    "adaptive_filter_length": 8,    # Shorter filter for lower CPU usage
}

# Example 4: Maximum quality configuration (when RNNoise is available)
max_quality_audio_settings = {
    "denoising_enabled": True,
    "prefer_rnnoise": True,         # Prefer RNNoise over fallback
    "noise_gate_threshold": 0.002,  # More aggressive noise gating
    "adaptive_filter_length": 32,   # Longer filter for better quality
}

# Example 5: Disabled denoising
disabled_audio_settings = {
    "denoising_enabled": False,
}

# Example usage in assistant creation
def create_assistant_with_denoising():
    """
    Example function showing how to create an assistant with audio denoising.
    """
    from app.models.assistant import Assistant
    
    assistant = Assistant(
        name="Customer Service Bot",
        phone_number="+1234567890",
        # ... other assistant settings ...
        
        # Enable audio denoising
        audio_settings=basic_audio_settings,
        
        # Alternative: use advanced settings
        # audio_settings=advanced_audio_settings,
    )
    
    return assistant

# Example usage in environment configuration
def configure_environment_defaults():
    """
    Example of setting environment variables for audio denoising defaults.
    """
    import os
    
    # Enable denoising by default for all assistants
    os.environ["AUDIO_DENOISING_DEFAULT"] = "true"
    
    # Force specific denoising method
    os.environ["AUDIO_DENOISING_METHOD"] = "auto"  # or "rnnoise" or "fallback"
    
    # Set default noise gate threshold
    os.environ["AUDIO_NOISE_GATE_THRESHOLD"] = "0.005"

# Example monitoring function
async def monitor_denoising_performance(call_handler, call_sid):
    """
    Example function to monitor denoising performance for a call.
    """
    call_state = call_handler.get_call_state(call_sid)
    
    if call_state and call_state.audio_denoising_service:
        stats = call_state.audio_denoising_service.get_stats()
        
        print(f"Call {call_sid} denoising stats:")
        print(f"  Method: {stats['method']}")
        print(f"  Enabled: {stats['enabled']}")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Avg processing time: {stats['avg_processing_time_ms']:.2f}ms")
        
        # Alert if processing time is too high
        if stats['avg_processing_time_ms'] > 5.0:
            print(f"WARNING: High denoising latency for call {call_sid}")
        
        return stats
    
    return None

# Example health check
def check_denoising_health():
    """
    Example health check for audio denoising system.
    """
    from app.services.audio_denoising_service import check_rnnoise_availability
    
    health_status = {
        "rnnoise_available": check_rnnoise_availability(),
        "fallback_available": True,  # Always available
    }
    
    if health_status["rnnoise_available"]:
        print("✅ RNNoise is available - using high-quality denoising")
    else:
        print("⚠️  RNNoise not available - using fallback filters")
    
    return health_status

if __name__ == "__main__":
    # Run health check
    print("Audio Denoising Health Check:")
    health = check_denoising_health()
    print(f"Health status: {health}")
    
    # Test configuration
    print("\nTesting audio denoising configuration...")
    
    # This would normally be done when creating an assistant
    print("Basic config:", basic_audio_settings)
    print("Advanced config:", advanced_audio_settings)
    print("High-volume config:", high_volume_audio_settings)
    print("Max quality config:", max_quality_audio_settings) 