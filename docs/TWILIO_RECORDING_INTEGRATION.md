# Twilio Recording Integration

This document explains how Twilio call recordings are integrated with the existing S3 recording system.

## Overview

The system now supports **dual recording** functionality:
1. **S3 Recording Service** - Real-time recording during the call using our custom recording service
2. **Twilio Native Recording** - High-quality recording managed by Twilio's infrastructure

Both recordings are stored in S3 and accessible through the same interface.

## How It Works

### 1. Call Start (TwiML Generation)
When a call starts, the `/twiml` endpoint:
- Generates TwiML with a `<Record>` verb to start Twilio recording
- Configures recording with:
  - Dual-channel recording (separate user and assistant tracks)
  - Recording status callback URL: `/recording/status`
  - High-quality settings (trim silence, no beep)
  - Maximum duration: 1 hour

```xml
<Response>
  <Record 
    recordingChannels="dual"
    recordingStatusCallback="https://yourdomain.com/recording/status"
    recordingStatusCallbackEvent="completed,failed"
    maxLength="3600"
    trim="trim-silence"
    playBeep="false" />
  <Connect>
    <Stream url="wss://yourdomain.com/streams" />
  </Connect>
</Response>
```

### 2. S3 Recording During Call
When the call handler starts:
- S3 Recording Service creates real-time recordings (user, assistant, mixed)
- No placeholder records needed - recordings are created as they happen

### 3. Recording Completion Webhook
When Twilio completes the recording, it calls `/recording/status`:
- Downloads the completed recording from Twilio
- Uploads it to S3 using the existing S3 service
- **Updates the existing "mixed" recording** with superior Twilio data:
  - Overwrites S3 key and URL with Twilio recording
  - Updates duration and file size from Twilio
  - Adds Recording SID and metadata
  - Changes status to `"completed"`
- If no "mixed" recording exists, creates a new one (fallback for disabled S3 recording)

## Database Schema

The `Recording` model supports both recording types:

```sql
-- Recording types
recording_type: 'user' | 'assistant' | 'mixed' | 'twilio'

-- Recording sources  
recording_source: 's3' | 'twilio' (deprecated)

-- Both local and Twilio recordings use 's3' as source
-- and are differentiated by recording_type
```

## API Endpoints

### TwiML Endpoint
- **URL**: `POST /twiml`
- **Purpose**: Generate TwiML with recording enabled
- **Twilio Features**: 
  - Dual-channel recording
  - Automatic recording start
  - Status callback configuration

### Recording Webhook
- **URL**: `POST /recording/status`
- **Purpose**: Handle Twilio recording completion
- **Process**:
  1. Receive recording metadata from Twilio
  2. Download recording content
  3. Upload to S3
  4. Update database record

### Recording Access
- **Download**: `GET /calls/{call_id}/recording/{recording_id}`
- **Play**: `GET /calls/{call_id}/recording/{recording_id}/play`
- **Support**: Both S3 local and Twilio recordings

## Recording Types

| Type | Source | Description |
|------|--------|-------------|
| `user` | S3 Recording Service | User audio only (real-time) |
| `assistant` | S3 Recording Service | Assistant audio only (real-time) |
| `mixed` | S3 + Twilio | Combined audio - starts as S3 real-time, upgraded to Twilio quality when available |

**Note**: The `mixed` recording provides the best of both worlds - real-time availability during the call via S3, then upgraded to high-quality Twilio recording when the call completes.

## Configuration

### Assistant Recording Settings
```json
{
  "recording_settings": {
    "enabled": true,  // Enables S3 recording service
    "format": "mp3",
    "record_user_audio": true,
    "record_assistant_audio": true,
    "record_mixed_audio": true
  }
}
```

### Twilio Recording
- **Enabled**: Automatically for all calls
- **Format**: MP3 (Twilio default)
- **Channels**: Dual-channel (2)
- **Quality**: High (Twilio managed)

## Benefits

### Reliability
- **Redundancy**: Two independent recording systems
- **Fallback**: If one system fails, the other continues
- **Quality**: Twilio provides professional-grade recording

### Features
- **Dual Channel**: Separate tracks for user and assistant
- **High Quality**: Twilio's optimized recording infrastructure
- **Automatic**: No additional configuration required
- **Integrated**: Same download/playback interface

### Storage
- **Unified**: All recordings stored in S3
- **Consistent**: Same database schema and API
- **Efficient**: Only stores completed recordings

## Technical Details

### TwiML Configuration
```python
# Recording configuration in /twiml endpoint
response.record(
    recording_channels="dual",
    recording_status_callback=recording_status_callback_url,
    recording_status_callback_event=["completed", "failed"],
    max_length=3600,
    trim="trim-silence",
    play_beep=False
)
```

### S3 Integration
```python
# Upload Twilio recording to S3 and update existing mixed recording
s3_key, s3_url = await s3_service.upload_audio_file(
    audio_data=recording_content,
    call_sid=call_sid,
    recording_type="twilio",  # This gets stored as metadata, not recording_type
    format="mp3",
    metadata=metadata,
)

# Update existing "mixed" recording with superior Twilio data
await ConversationService.update_recording(
    recording_id=mixed_recording.id,
    recording_sid=recording_sid,
    s3_key=s3_key,
    s3_url=s3_url,
    duration=duration,
    file_size=file_size,
    status="completed",
    recording_metadata={
        **metadata,
        "source": "twilio_override",
        "original_source": "s3"
    }
)
```

### Database Update
```python
# Update placeholder record with Twilio data
await ConversationService.update_recording(
    recording_id=twilio_recording.id,
    recording_sid=recording_sid,
    s3_key=s3_key,
    s3_url=s3_url,
    duration=duration,
    file_size=file_size,
    status="completed",
    recording_metadata=metadata,
)
```

## Error Handling

### Webhook Failures
- Returns 200 to Twilio to prevent retries
- Logs errors for monitoring
- Does not affect call functionality

### Download Failures
- Logs detailed error information
- Falls back gracefully
- S3 recording service continues independently

### Missing Credentials
- Validates Twilio credentials before download
- Uses assistant-specific or environment credentials
- Fails gracefully with proper error messages

## Monitoring

### Logging
- Recording start events
- Webhook reception and processing
- S3 upload success/failure
- Database record updates

### Metrics
- Recording completion rates
- Download success rates
- S3 upload performance
- Webhook response times

## Migration

### Existing Calls
- No changes required for existing functionality
- New calls automatically get Twilio recording
- Backward compatibility maintained

### Database
- No schema changes required
- Existing recording endpoints work unchanged
- New recording type seamlessly integrated

## Testing

To test the integration:

1. **Make a test call** to a configured assistant
2. **Check TwiML response** includes `<Record>` verb
3. **Verify S3 mixed recording creation** during call
4. **Wait for recording completion** (after call ends)
5. **Check webhook logs** for `/recording/status` calls
6. **Verify mixed recording updated** with Twilio data and S3 upload
7. **Test playback** through web interface

## Troubleshooting

### Common Issues

**Recording not starting**:
- Check TwiML includes `<Record>` verb
- Verify webhook URL is accessible
- Check Twilio account permissions

**Webhook not received**:
- Verify webhook URL accessibility
- Check Twilio webhook configuration
- Review firewall/proxy settings

**Download failures**:
- Verify Twilio credentials
- Check recording SID validity
- Review network connectivity

**S3 upload issues**:
- Check S3 service configuration
- Verify bucket permissions
- Review S3 credentials

### Debug Steps

1. **Check TwiML output**: Verify `<Record>` verb present
2. **Monitor webhook calls**: Check `/recording/status` endpoint
3. **Review logs**: Look for recording-related log messages
4. **Test S3 connectivity**: Verify S3 service functionality
5. **Validate credentials**: Check Twilio account access

## Future Enhancements

### Planned Features
- **Recording quality settings**: Configurable quality/compression
- **Recording archival**: Automatic archival of old recordings  
- **Recording analytics**: Enhanced metrics and reporting
- **Recording transcription**: Automatic transcription of Twilio recordings

### Integration Opportunities
- **AI analysis**: Sentiment analysis on dual-channel recordings
- **Quality monitoring**: Automated quality scoring
- **Compliance features**: PCI/HIPAA compliant recording options
- **Real-time processing**: Live analysis during recording 