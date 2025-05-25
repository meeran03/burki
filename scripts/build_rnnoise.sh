#!/bin/bash

# Build RNNoise as WebAssembly module for real-time audio denoising
# This script compiles RNNoise to WASM for use in the audio denoising service
#
# NOTE: For Railway deployment, RNNoise is automatically built in the Dockerfile.
# This script is only needed for local development or manual installation.

set -e

echo "Building RNNoise WebAssembly module..."

# Create build directory
BUILD_DIR="build/rnnoise"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Check if Emscripten is available
if ! command -v emcc &> /dev/null; then
    echo "Error: Emscripten not found. Please install Emscripten SDK first."
    echo "Visit: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

# Clone RNNoise if not already present
if [ ! -d "rnnoise" ]; then
    echo "Cloning RNNoise repository..."
    git clone https://github.com/xiph/rnnoise.git
fi

cd rnnoise

# Configure and build RNNoise
echo "Configuring RNNoise..."
./autogen.sh
emconfigure ./configure --disable-examples --disable-doc

echo "Building RNNoise..."
emmake make

# Compile to WebAssembly
echo "Compiling to WebAssembly..."

# Create the WASM module with exported functions
emcc -Os -g2 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MALLOC=emmalloc \
    -s MODULARIZE=1 \
    -s ENVIRONMENT="web,worker" \
    -s EXPORT_ES6=1 \
    -s USE_ES6_IMPORT_META=1 \
    -s WASM_ASYNC_COMPILATION=0 \
    -s SINGLE_FILE=1 \
    -s EXPORT_NAME=createRNNoiseModule \
    -s EXPORTED_FUNCTIONS="['_rnnoise_process_frame', '_rnnoise_create', '_rnnoise_destroy', '_malloc', '_free']" \
    -s EXPORTED_RUNTIME_METHODS="['ccall', 'cwrap']" \
    .libs/librnnoise.a \
    -o ../../static/rnnoise.js

echo "RNNoise WebAssembly module built successfully!"
echo "Output files:"
echo "  - static/rnnoise.js (WASM module with JS wrapper)"

# Create a simple test to verify the module works
cat > ../../static/test_rnnoise.js << 'EOF'
// Simple test for RNNoise WASM module
import createRNNoiseModule from './rnnoise.js';

async function testRNNoise() {
    try {
        const Module = await createRNNoiseModule();
        
        // Test basic functionality
        const rnnoise_create = Module.cwrap('rnnoise_create', 'number', []);
        const rnnoise_destroy = Module.cwrap('rnnoise_destroy', 'void', ['number']);
        const rnnoise_process_frame = Module.cwrap('rnnoise_process_frame', 'number', ['number', 'number']);
        
        // Create RNNoise state
        const state = rnnoise_create();
        console.log('RNNoise state created:', state);
        
        // Clean up
        rnnoise_destroy(state);
        console.log('RNNoise test completed successfully!');
        
        return true;
    } catch (error) {
        console.error('RNNoise test failed:', error);
        return false;
    }
}

// Export for use in Node.js or browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { testRNNoise };
} else {
    window.testRNNoise = testRNNoise;
}
EOF

echo "Test file created: static/test_rnnoise.js"
echo ""
echo "To test the module, run:"
echo "  node static/test_rnnoise.js"
echo ""
echo "Build complete!" 