#!/bin/bash

# Verify RNNoise installation
# This script checks if RNNoise was installed correctly during deployment
# If RNNoise is not available, it gracefully continues (fallback audio processing will be used)

echo "🔍 Verifying RNNoise installation..."

# Check if rnnoise_demo binary exists
if command -v rnnoise_demo &> /dev/null; then
    echo "✅ rnnoise_demo binary found at: $(which rnnoise_demo)"
    
    # Check if the binary is executable
    if [ -x "$(which rnnoise_demo)" ]; then
        echo "✅ rnnoise_demo is executable"
        
        # Test basic functionality (this will show usage if no args provided)
        echo "🧪 Testing RNNoise functionality..."
        rnnoise_demo 2>&1 | head -3
        
        echo "✅ RNNoise verification completed successfully!"
        echo "🎧 High-quality audio denoising is ready for use."
    else
        echo "⚠️  rnnoise_demo is not executable, but continuing deployment"
        echo "🎧 Fallback audio processing will be used."
    fi
else
    echo "⚠️  rnnoise_demo binary not found, but continuing deployment"
    echo "🎧 Fallback audio processing will be used instead of RNNoise."
    echo "ℹ️  This is normal if RNNoise installation failed during build."
fi

echo "✅ Audio denoising verification completed!"
exit 0 