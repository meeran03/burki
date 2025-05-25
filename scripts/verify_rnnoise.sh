#!/bin/bash

# Verify RNNoise installation
# This script checks if RNNoise was installed correctly during deployment

echo "🔍 Verifying RNNoise installation..."

# Check if rnnoise_demo binary exists
if command -v rnnoise_demo &> /dev/null; then
    echo "✅ rnnoise_demo binary found at: $(which rnnoise_demo)"
else
    echo "❌ rnnoise_demo binary not found"
    exit 1
fi

# Check if the binary is executable
if [ -x "$(which rnnoise_demo)" ]; then
    echo "✅ rnnoise_demo is executable"
else
    echo "❌ rnnoise_demo is not executable"
    exit 1
fi

# Test basic functionality (this will show usage if no args provided)
echo "🧪 Testing RNNoise functionality..."
rnnoise_demo 2>&1 | head -3

echo "✅ RNNoise verification completed successfully!"
echo "🎧 Audio denoising is ready for use." 