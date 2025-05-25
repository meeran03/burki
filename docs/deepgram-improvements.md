# Deepgram Integration Improvements

This document outlines the improvements made to better follow Deepgram's best practices for speech-to-text transcription, particularly around endpointing and interim results.

## Key Changes Made

### 1. **Proper `speech_final` Handling**

Previously, the system only processed transcripts when `is_final` was true, completely ignoring `speech_final`. Now:

- **`speech_final: true`** is used to detect natural speech boundaries (pauses in conversation)
- **`is_final: true`** is used for transcript accuracy (Deepgram's finalized transcript)
- Complete utterances are formed by accumulating `is_final` transcripts until `speech_final` is reached

### 2. **Utterance Accumulation**

Following Deepgram's guidelines, we now:

- Accumulate all `is_final: true` transcripts in an utterance buffer
- When `speech_final: true` is received, concatenate all buffered transcripts for the complete utterance
- Process the complete utterance with the LLM for better context understanding

### 3. **Default Endpointing Value**

- Changed default endpointing from 100ms to 10ms (Deepgram's default)
- This enables faster detection of speech pauses
- Particularly useful for quick responses like "yes" or "no"

### 4. **Improved Interruption Detection**

- Interruption detection now works with complete utterances rather than individual transcript segments
- Better handles natural speech patterns and pauses

## Benefits

1. **Lower Latency**: Using `speech_final` allows the system to react quickly to short utterances
2. **Better Context**: Complete utterances provide more context to the LLM
3. **More Natural Conversations**: Respects natural speech boundaries instead of arbitrary timing
4. **Improved Accuracy**: Accumulating transcripts until speech_final ensures complete thoughts are captured

## Configuration Options

Assistants can still customize these settings through `stt_settings`:

```json
{
  "stt_settings": {
    "endpointing": {
      "silence_threshold": 10  // ms of silence to detect endpoint (default: 10)
    },
    "utterance_end_ms": 1000,  // ms to wait before ending utterance
    "interim_results": true,    // whether to receive interim results
    "process_interim_results": false  // optional: process interim results for ultra-low latency
  }
}
```

## Code Example

Here's how the improved transcript handling works:

```python
# When is_final=true, accumulate transcript
if is_final and transcript.strip():
    utterance_buffer.append(transcript.strip())

# When speech_final=true, process complete utterance
if speech_final and utterance_buffer:
    complete_utterance = " ".join(utterance_buffer)
    # Process complete utterance with LLM
    utterance_buffer.clear()
```

## Migration Notes

- Existing assistants will automatically benefit from these improvements
- No configuration changes required unless you want to adjust endpointing sensitivity
- The system maintains backward compatibility with existing settings

## LLM Request Cancellation (Multiple Response Fix)

### Problem
When users interrupted the AI assistant, multiple LLM responses would be generated and spoken, creating a confusing experience. This happened because:

1. The original LLM request would continue processing even after an interruption
2. The interruption would trigger a new LLM request
3. Any buffered transcripts would trigger additional LLM requests
4. All these responses would be spoken, overlapping each other

### Solution
We've implemented proper LLM request tracking and cancellation:

1. **Request Tracking**: Each LLM request is now tracked as an asyncio Task in `pending_llm_task`
2. **Interruption Handling**: When an interruption is detected, any pending LLM task is cancelled before processing the new request
3. **Sequential Processing**: New utterances cancel any pending LLM requests to ensure only one response is generated at a time
4. **Clean Shutdown**: When a call ends, any pending LLM tasks are properly cancelled

### Implementation Details

```python
# Track LLM requests
self.active_calls[call_sid].pending_llm_task = asyncio.create_task(
    llm_service.process_transcript(...)
)

# Cancel on interruption
if pending_llm_task and not pending_llm_task.done():
    pending_llm_task.cancel()
    try:
        await asyncio.wait_for(pending_llm_task, timeout=0.1)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass  # Expected
```

This ensures that users hear only the most relevant response when they interrupt or speak multiple utterances quickly.

## UtteranceEnd Support for Noisy Environments

### Problem
According to [Deepgram's documentation](https://developers.deepgram.com/docs/understanding-end-of-speech-detection), endpointing alone can fail in noisy environments:

> "In environments with significant background noise such as playing music, a ringing phone, or at a fast food drive thru, the background noise can cause the VAD to trigger and prevent the detection of silent audio. Since endpointing only fires after a certain amount of silence has been detected, a significant amount of background noise may prevent the `speech_final=true` flag from being sent."

### Solution
We've implemented **UtteranceEnd** as a fallback mechanism that works even in noisy environments:

1. **Word-based Detection**: UtteranceEnd analyzes word timings instead of audio silence
2. **Noise Immunity**: Works regardless of background noise levels
3. **Fallback Processing**: Triggers when `speech_final` fails to fire
4. **Configurable Timing**: Uses 1000ms gap detection by default (recommended minimum)

### Implementation Details

```python
# UtteranceEnd configuration
{
  "stt_settings": {
    "utterance_end_ms": 1000,  // Gap in ms between words to trigger UtteranceEnd
    "vad_events": true,        // Enable VAD events for better detection
    "interim_results": true    // Required for UtteranceEnd to work
  }
}
```

### How It Works

1. **Primary Path**: System tries to use `speech_final` for natural speech boundaries
2. **Fallback Path**: If `speech_final` doesn't fire due to noise, UtteranceEnd triggers based on word gaps
3. **Dual Processing**: Both mechanisms can work together for maximum reliability

This ensures reliable speech detection in all environments, from quiet offices to noisy restaurants or drive-throughs.

## Architecture Compliance

Our implementation now fully complies with Deepgram's recommendations:

✅ **Endpointing**: Proper `speech_final` handling with 10ms default  
✅ **Interim Results**: Correct utterance accumulation until `speech_final`  
✅ **UtteranceEnd**: Fallback detection for noisy environments  
✅ **VAD Events**: Enhanced voice activity detection  
✅ **Request Management**: Proper LLM task cancellation  
✅ **Timing Heuristics**: Word-gap based utterance segmentation 