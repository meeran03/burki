# Railway Deployment Guide

This guide covers deploying Burqi to Railway with audio denoising support.

## Overview

The Dockerfile has been configured to automatically install RNNoise during deployment, enabling high-quality audio denoising for your voice AI assistants.

## What Gets Installed

During Railway deployment, the following audio processing components are automatically installed:

### System Dependencies
- `gcc`, `g++` - Compilers for building RNNoise
- `autoconf`, `automake`, `libtool` - Build tools
- `git` - For cloning RNNoise source code
- `portaudio19-dev` - Audio processing libraries

### RNNoise Installation
- Clones RNNoise from the official Xiph repository
- Compiles from source with optimizations
- Installs the `rnnoise_demo` binary to `/usr/local/bin/`
- Automatically verifies the installation

## Deployment Steps

1. **Connect Repository**: Connect your GitHub repository to Railway
2. **Environment Variables**: Set your required environment variables in Railway dashboard
3. **Deploy**: Railway will automatically build using the Dockerfile
4. **Verify**: Check the build logs for RNNoise verification messages

## Build Process

The Dockerfile performs these steps in order:

```dockerfile
# 1. Install system dependencies (including build tools)
RUN apt-get update && apt-get install -y gcc g++ git autoconf automake libtool make ...

# 2. Build and install RNNoise
RUN mkdir -p /tmp/rnnoise && cd /tmp/rnnoise \
    && git clone https://github.com/xiph/rnnoise.git \
    && cd rnnoise \
    && ./autogen.sh \
    && ./configure --prefix=/usr/local \
    && make && make install \
    && cp examples/.libs/rnnoise_demo /usr/local/bin/

# 3. Install Python dependencies
RUN pip install -r requirements.txt

# 4. Verify RNNoise installation
RUN /app/scripts/verify_rnnoise.sh
```

## Expected Build Output

During deployment, you should see output like:

```
🔍 Verifying RNNoise installation...
✅ rnnoise_demo binary found at: /usr/local/bin/rnnoise_demo
✅ rnnoise_demo is executable
🧪 Testing RNNoise functionality...
Usage: rnnoise_demo <input.raw> <output.raw>
✅ RNNoise verification completed successfully!
🎧 Audio denoising is ready for use.
```

## Enabling Audio Denoising

After deployment:

1. Go to your assistant settings in the web interface
2. Navigate to the **Speech-to-Text** tab
3. Check the **Audio Denoising** checkbox
4. Save the assistant

## Troubleshooting

### Build Fails During RNNoise Installation

If the build fails during RNNoise installation:

1. Check Railway build logs for specific error messages
2. Common issues:
   - Missing system dependencies (should be handled by Dockerfile)
   - Network issues during git clone (retry deployment)
   - Compilation errors (check Railway's build environment)

### RNNoise Verification Fails

If the verification step fails:

1. The build will stop and show an error
2. Check if all system dependencies were installed correctly
3. Verify the RNNoise compilation completed successfully

### Runtime Issues

If audio denoising doesn't work at runtime:

1. Check application logs for RNNoise-related errors
2. Verify the `rnnoise_demo` binary is accessible
3. Ensure the assistant has audio denoising enabled in settings

## Performance Considerations

### Build Time
- RNNoise compilation adds ~2-3 minutes to build time
- This is a one-time cost during deployment
- Subsequent deployments use Railway's layer caching

### Runtime Performance
- RNNoise processing is very lightweight (~1-2ms per frame)
- Minimal impact on call latency
- Uses CPU efficiently with optimized algorithms

### Memory Usage
- RNNoise has minimal memory footprint (~1MB)
- No significant impact on container memory usage

## Environment Variables

No additional environment variables are required for RNNoise. The audio denoising feature is controlled through the assistant settings in the web interface.

## Monitoring

You can monitor audio denoising performance through:

1. **Application Logs**: Look for denoising-related log messages
2. **Call Quality**: Monitor transcription accuracy improvements
3. **Performance Metrics**: Check for any latency impacts

## Support

If you encounter issues with audio denoising on Railway:

1. Check the build logs for RNNoise installation errors
2. Verify your assistant settings have audio denoising enabled
3. Test with a simple call to ensure the feature is working

The audio denoising feature is designed to be robust and fail gracefully - if RNNoise isn't available, calls will continue to work normally without denoising. 