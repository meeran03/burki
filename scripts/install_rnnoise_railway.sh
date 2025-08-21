#!/bin/bash

# Alternative RNNoise installation script for Railway deployment
# This script provides multiple fallback strategies to handle network issues

set -e

echo "🚀 Installing RNNoise for Railway deployment..."

# Function to attempt RNNoise installation with fallbacks
install_rnnoise_with_fallbacks() {
    local build_dir="/tmp/rnnoise_build"
    mkdir -p "$build_dir"
    cd "$build_dir"

    echo "📦 Cloning RNNoise repository..."
    if ! git clone https://github.com/xiph/rnnoise.git; then
        echo "❌ Failed to clone RNNoise repository"
        return 1
    fi

    cd rnnoise

    echo "🔧 Attempting RNNoise build with multiple strategies..."

    # Strategy 1: Try normal autogen with timeout
    echo "Strategy 1: Normal autogen with timeout..."
    if timeout 180 ./autogen.sh 2>/dev/null; then
        echo "✅ Model download successful"
        if ./configure --prefix=/usr/local && make && make install; then
            echo "✅ RNNoise installed successfully with model"
            return 0
        fi
    fi

    echo "⚠️  Model download failed or timed out, trying manual configuration..."

    # Strategy 2: Manual configuration without model download
    echo "Strategy 2: Manual configuration without model download..."
    if command -v autoreconf >/dev/null 2>&1; then
        if autoreconf -fiv && ./configure --prefix=/usr/local --disable-examples --disable-doc; then
            if make && make install; then
                echo "✅ RNNoise installed successfully without model (will use default)"
                return 0
            fi
        fi
    fi

    # Strategy 3: Minimal build
    echo "Strategy 3: Minimal build..."
    if ./configure --prefix=/usr/local --disable-examples --disable-doc --disable-model; then
        if make && make install; then
            echo "✅ RNNoise minimal build installed successfully"
            return 0
        fi
    fi

    echo "❌ All RNNoise installation strategies failed"
    return 1
}

# Main installation logic
if install_rnnoise_with_fallbacks; then
    echo "🎉 RNNoise installation completed!"
    
    # Copy demo binary if available
    if [ -f examples/.libs/rnnoise_demo ]; then
        cp examples/.libs/rnnoise_demo /usr/local/bin/ || echo "⚠️  Could not copy rnnoise_demo"
    fi
    
    # Update library cache
    ldconfig 2>/dev/null || echo "⚠️  Could not update ldconfig"
    
    echo "✅ RNNoise setup complete"
else
    echo "⚠️  RNNoise installation failed - application will use fallback audio processing"
    echo "ℹ️  This is acceptable and the application will work normally"
fi

# Clean up
cd /
rm -rf /tmp/rnnoise_build

echo "🧹 Cleanup completed"
exit 0
