import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import httpx
from openai import AsyncOpenAI

from app.db.models import Call, Assistant, Recording, WebhookLog
from app.services.conversation_service import ConversationService
from app.utils.url_utils import get_server_base_url

logger = logging.getLogger(__name__)


class WebhookService:
    """
    Service for sending webhooks to external endpoints during call lifecycle events.
    """

    # Track calls that have already sent end-of-call webhooks to prevent duplicates
    _webhooks_sent: Set[str] = set()

    # Locks to prevent race conditions when sending webhooks
    _webhook_locks: Dict[str, asyncio.Lock] = {}

    @staticmethod
    async def _get_or_create_lock(call_sid: str) -> asyncio.Lock:
        """Get or create a lock for a specific call to prevent duplicate webhooks."""
        if call_sid not in WebhookService._webhook_locks:
            WebhookService._webhook_locks[call_sid] = asyncio.Lock()
        return WebhookService._webhook_locks[call_sid]

    @staticmethod
    async def send_status_update_webhook(
        assistant: Assistant,
        call: Call,
        status: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        """
        Send a status update webhook for call events.

        Args:
            assistant: The assistant handling the call
            call: The call object
            status: The current status (in-progress, completed, failed)
            messages: Optional conversation messages

        Returns:
            bool: True if webhook was sent successfully, False otherwise
        """
        if not assistant.webhook_url:
            logger.debug(f"No webhook URL configured for assistant {assistant.id}")
            return True  # Not an error if no webhook is configured

        start_time = datetime.now()
        try:
            payload = {
                "message": {
                    "type": "status-update",
                    "status": status,
                    "phoneNumber": {"number": call.to_phone_number},
                    "call": {
                        "id": call.call_sid,
                        "phoneCallProviderId": call.call_sid,
                        "customer": {"number": call.customer_phone_number},
                    },
                    "messages": messages or [],
                }
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    assistant.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                # Calculate response time
                response_time_ms = int(
                    (datetime.now() - start_time).total_seconds() * 1000
                )

                success = response.status_code in [200, 201, 202]
                response_headers = dict(response.headers) if response.headers else None

                # Log the webhook attempt
                await WebhookService.log_webhook_attempt(
                    call_id=call.id,
                    assistant_id=assistant.id,
                    webhook_url=assistant.webhook_url,
                    webhook_type="status-update",
                    request_payload=payload,
                    response_status_code=response.status_code,
                    response_body=response.text,
                    response_headers=response_headers,
                    response_time_ms=response_time_ms,
                    success=success,
                    metadata={"status": status},
                )

                if success:
                    logger.info(
                        f"Status update webhook sent successfully for call {call.call_sid}"
                    )
                    return True
                else:
                    logger.warning(
                        f"Webhook failed with status {response.status_code} for call {call.call_sid}: {response.text}"
                    )
                    return False

        except Exception as e:
            # Calculate response time even for errors
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Log the failed webhook attempt
            await WebhookService.log_webhook_attempt(
                call_id=call.id,
                assistant_id=assistant.id,
                webhook_url=assistant.webhook_url,
                webhook_type="status-update",
                request_payload=payload,
                response_time_ms=response_time_ms,
                success=False,
                error_message=str(e),
                metadata={"status": status},
            )

            logger.error(
                f"Error sending status update webhook for call {call.call_sid}: {e}"
            )
            return False

    @staticmethod
    async def send_end_of_call_webhook(
        assistant: Assistant,
        call: Call,
        ended_reason: str = "customer-ended-call",
        recording_url: Optional[str] = None,
    ) -> bool:
        """
        Send an end-of-call report webhook with analysis and structured data.

        Args:
            assistant: The assistant that handled the call
            call: The completed call object
            ended_reason: Reason why the call ended
            recording_url: URL to the call recording

        Returns:
            bool: True if webhook was sent successfully, False otherwise
        """
        if not assistant.webhook_url:
            logger.debug(f"No webhook URL configured for assistant {assistant.id}")
            return True  # Not an error if no webhook is configured

        # Use a lock to prevent duplicate webhook sending for the same call
        lock = await WebhookService._get_or_create_lock(call.call_sid)

        async with lock:
            # Double-check if webhook has already been sent (inside the lock)
            if call.call_sid in WebhookService._webhooks_sent:
                logger.info(
                    f"End-of-call webhook already sent for call {call.call_sid}, skipping"
                )
                return True

            # Mark webhook as being sent immediately to prevent duplicates
            WebhookService._webhooks_sent.add(call.call_sid)

            try:
                # Get conversation messages
                messages = await WebhookService._get_conversation_messages(
                    call.call_sid
                )

                # Generate summary using OpenAI
                summary = await WebhookService._generate_call_summary(
                    assistant, messages
                )

                # Generate structured data based on assistant's schema
                structured_data = await WebhookService._generate_structured_data(
                    assistant, messages, summary
                )

                # Calculate duration
                duration_seconds = call.duration or 0
                if call.started_at and call.ended_at:
                    duration_seconds = int(
                        (call.ended_at - call.started_at).total_seconds()
                    )

                payload = {
                    "message": {
                        "type": "end-of-call-report",
                        "phoneNumber": {"number": call.to_phone_number},
                        "customer": {"number": call.customer_phone_number},
                        "summary": summary,
                        "recordingUrl": recording_url,
                        "endedReason": ended_reason,
                        "durationSeconds": duration_seconds,
                        "messages": messages,
                        "analysis": {
                            "summary": summary,
                            "durationSeconds": duration_seconds,
                            "structuredData": structured_data,
                        },
                    }
                }

                start_time = datetime.now()
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        assistant.webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )

                    # Calculate response time
                    response_time_ms = int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    )

                    success = response.status_code in [200, 201, 202]
                    response_headers = (
                        dict(response.headers) if response.headers else None
                    )

                    # Log the webhook attempt
                    await WebhookService.log_webhook_attempt(
                        call_id=call.id,
                        assistant_id=assistant.id,
                        webhook_url=assistant.webhook_url,
                        webhook_type="end-of-call-report",
                        request_payload=payload,
                        response_status_code=response.status_code,
                        response_body=response.text,
                        response_headers=response_headers,
                        response_time_ms=response_time_ms,
                        success=success,
                        metadata={
                            "ended_reason": ended_reason,
                            "duration_seconds": duration_seconds,
                            "recording_url": recording_url,
                        },
                    )

                    if success:
                        logger.info(
                            f"End-of-call webhook sent successfully for call {call.call_sid}"
                        )
                        return True
                    else:
                        logger.warning(
                            f"End-of-call webhook failed with status {response.status_code} for call {call.call_sid}: {response.text}"
                        )
                        # Remove from sent set since it failed
                        WebhookService._webhooks_sent.discard(call.call_sid)
                        return False

            except Exception as e:
                # Calculate response time for errors if we have a start_time
                response_time_ms = None
                if "start_time" in locals():
                    response_time_ms = int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    )

                # Log the failed webhook attempt if we have the payload
                if "payload" in locals():
                    await WebhookService.log_webhook_attempt(
                        call_id=call.id,
                        assistant_id=assistant.id,
                        webhook_url=assistant.webhook_url,
                        webhook_type="end-of-call-report",
                        request_payload=payload,
                        response_time_ms=response_time_ms,
                        success=False,
                        error_message=str(e),
                        metadata={
                            "ended_reason": ended_reason,
                            "recording_url": recording_url,
                        },
                    )

                logger.error(
                    f"Error sending end-of-call webhook for call {call.call_sid}: {e}"
                )
                # Remove from sent set since it failed
                WebhookService._webhooks_sent.discard(call.call_sid)
                return False

    @staticmethod
    async def _get_conversation_messages(call_sid: str) -> List[Dict[str, str]]:
        """
        Get conversation messages from transcripts.

        Args:
            call_sid: The call SID

        Returns:
            List of message dictionaries with role and content
        """
        try:
            transcripts = await ConversationService.get_call_transcripts(call_sid)
            messages = []

            for transcript in transcripts:
                role = "bot" if transcript.speaker == "assistant" else "human"
                messages.append({"role": role, "content": transcript.content})

            return messages

        except Exception as e:
            logger.error(
                f"Error getting conversation messages for call {call_sid}: {e}"
            )
            return []

    @staticmethod
    async def _generate_call_summary(
        assistant: Assistant, messages: List[Dict[str, str]]
    ) -> str:
        """
        Generate a call summary using OpenAI GPT-4o-mini.

        Args:
            assistant: The assistant that handled the call
            messages: The conversation messages

        Returns:
            Generated summary text
        """
        try:
            # Use assistant's OpenAI API key if available, otherwise use environment variable
            api_key = assistant.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("No OpenAI API key available for summary generation")
                return "Summary generation unavailable - no API key configured"

            client = AsyncOpenAI(api_key=api_key)

            # Prepare conversation for summary
            conversation_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in messages]
            )

            if not conversation_text.strip():
                return "No conversation content to summarize"

            # Generate summary
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at summarizing phone conversations. Provide a concise, professional summary of the call that highlights the main topics discussed, any requests made, and the outcome. Keep it brief but informative.",
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize this phone conversation:\n\n{conversation_text}",
                    },
                ],
                max_tokens=200,
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating call summary: {e}")
            return f"Error generating summary: {str(e)}"

    @staticmethod
    async def _generate_structured_data(
        assistant: Assistant, messages: List[Dict[str, str]], summary: str
    ) -> Dict[str, Any]:
        """
        Generate structured data based on the assistant's configured schema using OpenAI structured outputs.

        Args:
            assistant: The assistant with potential schema configuration
            messages: The conversation messages
            summary: The call summary

        Returns:
            Dictionary with structured data extracted from the conversation
        """
        try:
            # Use assistant's OpenAI API key if available, otherwise use environment variable
            api_key = assistant.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "No OpenAI API key available for structured data generation"
                )
                return {}

            # Get structured data schema from assistant's custom settings
            schema = None
            if (
                assistant.custom_settings
                and "structured_data_schema" in assistant.custom_settings
            ):
                schema = assistant.custom_settings["structured_data_schema"]

            # Get custom structured data prompt from assistant's custom settings
            custom_prompt = None
            if (
                assistant.custom_settings
                and "structured_data_prompt" in assistant.custom_settings
            ):
                custom_prompt = assistant.custom_settings["structured_data_prompt"]

            # Default schema if none configured
            if not schema:
                schema = {
                    "type": "object",
                    "properties": {
                        "chat_topic": {
                            "type": "string",
                            "description": "The main topic of the conversation",
                        },
                        "followup_sms": {
                            "type": "string",
                            "description": "A follow-up SMS message to send to the customer",
                        },
                    },
                    "required": ["chat_topic"],
                }

            # Default structured data prompt if none configured
            if not custom_prompt:
                custom_prompt = f"""You are an expert at extracting structured data from phone conversations.

Analyze the conversation and extract information according to the configured schema.
Be accurate and only extract information that is clearly present in the conversation.

Call Summary: {summary}"""
            else:
                # Replace placeholders in custom prompt (only {summary} now, schema is handled by OpenAI)
                custom_prompt = custom_prompt.replace("{summary}", summary)
                # Remove any {schema} placeholders if they exist (for backward compatibility)
                if "{schema}" in custom_prompt:
                    custom_prompt = custom_prompt.replace(
                        "{schema}",
                        "[Schema is automatically handled by OpenAI structured outputs]",
                    )

            client = AsyncOpenAI(api_key=api_key)

            # Prepare conversation for analysis
            conversation_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in messages]
            )

            if not conversation_text.strip():
                return {}

            # Generate structured data using OpenAI's structured outputs
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": custom_prompt},
                    {
                        "role": "user",
                        "content": f"Extract structured data from this conversation:\n\n{conversation_text}",
                    },
                ],
                max_tokens=1000,
                temperature=0.8,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "structured_data", "schema": schema},
                },
            )

            # Parse the structured response
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            else:
                return {}

        except Exception as e:
            logger.error(f"Error generating structured data: {e}")
            return {"error": f"Failed to generate structured data: {str(e)}"}

    @staticmethod
    async def send_end_of_call_webhook_when_ready(
        call_sid: str, max_wait_seconds: int = 30
    ) -> bool:
        """
        Send end-of-call webhook after waiting for recordings to complete, or after timeout.

        Args:
            call_sid: The call SID
            max_wait_seconds: Maximum time to wait for recordings to complete

        Returns:
            bool: True if webhook was sent successfully, False otherwise
        """
        import asyncio
        from app.services.conversation_service import ConversationService

        try:
            # Check if webhook has already been sent for this call
            if call_sid in WebhookService._webhooks_sent:
                logger.info(
                    f"End-of-call webhook already sent for call {call_sid}, skipping delayed webhook"
                )
                return True

            # Get call and assistant info
            call = await ConversationService.get_conversation_by_sid(call_sid)
            if not call or not call.assistant:
                logger.error(f"Call or assistant not found for SID: {call_sid}")
                return False

            assistant = call.assistant

            # Wait for recordings to complete or timeout
            wait_time = 0
            check_interval = 2

            while wait_time < max_wait_seconds:
                # Check again if webhook was sent while we were waiting
                if call_sid in WebhookService._webhooks_sent:
                    logger.info(
                        f"End-of-call webhook was sent by another process for call {call_sid}, stopping wait"
                    )
                    return True

                recordings = await ConversationService.get_call_recordings(call_sid)

                # Check if all recordings are completed or failed
                all_recordings_done = True
                recording_url = None

                for recording in recordings:
                    if recording.status not in ["completed", "failed"]:
                        all_recordings_done = False
                        break
                    elif recording.status == "completed" and not recording_url:
                        # Use proper URL construction for the first completed recording
                        recording_url = WebhookService._construct_recording_url(
                            recording
                        )

                if all_recordings_done:
                    logger.info(
                        f"All recordings completed for call {call_sid}, sending webhook"
                    )
                    break

                # Wait before checking again
                await asyncio.sleep(check_interval)
                wait_time += check_interval

            if wait_time >= max_wait_seconds:
                logger.warning(
                    f"Timeout waiting for recordings to complete for call {call_sid}, sending webhook anyway"
                )

            # Get best available recording URL using proper URL construction
            recordings = await ConversationService.get_call_recordings(call_sid)
            recording_url = None
            if recordings:
                # Use the first completed recording, or any recording if none are completed
                completed_recordings = [
                    r for r in recordings if r.status == "completed"
                ]
                target_recording = (
                    completed_recordings[0] if completed_recordings else recordings[0]
                )
                recording_url = WebhookService._construct_recording_url(
                    target_recording
                )

            # Final check before sending webhook
            if call_sid in WebhookService._webhooks_sent:
                logger.info(
                    f"End-of-call webhook was sent by another process for call {call_sid}, skipping final send"
                )
                return True

            # Send the webhook
            return await WebhookService.send_end_of_call_webhook(
                assistant=assistant,
                call=call,
                ended_reason="customer-ended-call",
                recording_url=recording_url,
            )

        except Exception as e:
            logger.error(
                f"Error sending delayed end-of-call webhook for call {call_sid}: {e}"
            )
            return False

    @staticmethod
    async def send_end_of_call_webhook_with_recordings(
        call_sid: str, saved_files: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Send end-of-call webhook with S3 recording information.

        Args:
            call_sid: The call SID
            saved_files: Dictionary of recording type to S3 file info

        Returns:
            bool: True if webhook was sent successfully, False otherwise
        """
        try:
            # Get call and assistant info
            call = await ConversationService.get_conversation_by_sid(call_sid)
            if not call or not call.assistant:
                logger.error(f"Call or assistant not found for SID: {call_sid}")
                return False

            assistant = call.assistant

            # Get recording URL from database records instead of S3 file info
            recording_url = None
            if saved_files:
                # Get the recordings from the database to construct proper URLs
                recordings = await ConversationService.get_call_recordings(call_sid)
                if recordings:
                    # Prefer mixed recording, then user, then assistant
                    target_recording = None
                    for recording in recordings:
                        if recording.recording_type == "mixed":
                            target_recording = recording
                            break

                    if not target_recording:
                        for recording in recordings:
                            if recording.recording_type == "user":
                                target_recording = recording
                                break

                    if not target_recording:
                        for recording in recordings:
                            if recording.recording_type == "assistant":
                                target_recording = recording
                                break

                    if not target_recording and recordings:
                        target_recording = recordings[0]  # Use first available

                    if target_recording:
                        recording_url = WebhookService._construct_recording_url(
                            target_recording
                        )

            # Send the webhook
            return await WebhookService.send_end_of_call_webhook(
                assistant=assistant,
                call=call,
                ended_reason="customer-ended-call",
                recording_url=recording_url,
            )

        except Exception as e:
            logger.error(
                f"Error sending end-of-call webhook with recordings for call {call_sid}: {e}"
            )
            return False

    @staticmethod
    async def send_end_of_call_webhook_immediate(call_sid: str) -> bool:
        """
        Send end-of-call webhook immediately without waiting for recordings.

        Args:
            call_sid: The call SID

        Returns:
            bool: True if webhook was sent successfully, False otherwise
        """
        try:
            # Get call and assistant info
            call = await ConversationService.get_conversation_by_sid(call_sid)
            if not call or not call.assistant:
                logger.error(f"Call or assistant not found for SID: {call_sid}")
                return False

            assistant = call.assistant

            # Send the webhook without recording URL
            return await WebhookService.send_end_of_call_webhook(
                assistant=assistant,
                call=call,
                ended_reason="customer-ended-call",
                recording_url=None,
            )

        except Exception as e:
            logger.error(
                f"Error sending immediate end-of-call webhook for call {call_sid}: {e}"
            )
            return False

    @staticmethod
    def _construct_local_recording_url(call_sid: str, file_path: str) -> str:
        """
        Construct a URL for a local recording file.

        Args:
            call_sid: The call SID
            file_path: The local file path

        Returns:
            str: The constructed URL
        """
        try:
            # Get the base server URL
            base_url = get_server_base_url()

            # Extract filename from path
            import os

            filename = os.path.basename(file_path)

            # Construct URL for local recording endpoint
            # This assumes you have an endpoint to serve local recordings
            recording_url = f"{base_url}/recordings/{call_sid}/{filename}"

            return recording_url

        except Exception as e:
            logger.error(f"Error constructing local recording URL for {file_path}: {e}")
            return file_path  # Fallback to file path

    @staticmethod
    def cleanup_webhook_tracking(max_entries: int = 1000) -> None:
        """
        Clean up webhook tracking to prevent memory leaks.

        Args:
            max_entries: Maximum number of entries to keep in memory
        """
        # Clean up webhooks_sent set
        if len(WebhookService._webhooks_sent) > max_entries:
            # Remove oldest half of entries (simple approach)
            entries_to_remove = len(WebhookService._webhooks_sent) - max_entries // 2
            call_sids_to_remove = list(WebhookService._webhooks_sent)[
                :entries_to_remove
            ]
            for call_sid in call_sids_to_remove:
                WebhookService._webhooks_sent.discard(call_sid)
            logger.info(f"Cleaned up {entries_to_remove} webhook tracking entries")

        # Clean up webhook locks
        if len(WebhookService._webhook_locks) > max_entries:
            # Remove locks for calls that are no longer being tracked
            locks_to_remove = []
            for call_sid in WebhookService._webhook_locks:
                if call_sid not in WebhookService._webhooks_sent:
                    locks_to_remove.append(call_sid)

            for call_sid in locks_to_remove:
                WebhookService._webhook_locks.pop(call_sid, None)

            logger.info(f"Cleaned up {len(locks_to_remove)} webhook lock entries")

    @staticmethod
    def _construct_recording_url(recording: Recording) -> Optional[str]:
        """
        Construct the appropriate URL for a recording.

        Args:
            recording: Recording object

        Returns:
            Optional[str]: Recording URL or None
        """
        if recording.s3_url:
            return recording.s3_url
        elif recording.recording_sid:
            # Fallback to Twilio URL construction (deprecated)
            return f"https://api.twilio.com/2010-04-01/Accounts/{os.getenv('TWILIO_ACCOUNT_SID')}/Recordings/{recording.recording_sid}.mp3"
            return None

    @staticmethod
    async def log_webhook_attempt(
        call_id: int,
        assistant_id: int,
        webhook_url: str,
        webhook_type: str,
        request_payload: Dict[str, Any],
        response_status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        response_headers: Optional[Dict[str, str]] = None,
        response_time_ms: Optional[int] = None,
        success: bool = False,
        error_message: Optional[str] = None,
        retry_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[WebhookLog]:
        """
        Log a webhook attempt to the database.

        Args:
            call_id: Call ID
            assistant_id: Assistant ID
            webhook_url: Webhook URL
            webhook_type: Type of webhook (status-update, end-of-call-report)
            request_payload: The payload that was sent
            response_status_code: HTTP response status code
            response_body: Response body (will be truncated if too long)
            response_headers: Response headers
            response_time_ms: Response time in milliseconds
            success: Whether the webhook was successful
            error_message: Error message if failed
            retry_count: Number of retries attempted
            metadata: Additional metadata

        Returns:
            Optional[WebhookLog]: Created webhook log or None
        """
        try:
            from app.db.database import get_async_db_session

            async with await get_async_db_session() as db:
                # Truncate response body if too long (max 10KB)
                truncated_response_body = response_body
                if response_body and len(response_body) > 10240:
                    truncated_response_body = response_body[:10240] + "... [TRUNCATED]"

                webhook_log_data = {
                    "call_id": call_id,
                    "assistant_id": assistant_id,
                    "webhook_url": webhook_url,
                    "webhook_type": webhook_type,
                    "request_payload": request_payload,
                    "request_headers": {"Content-Type": "application/json"},
                    "response_status_code": response_status_code,
                    "response_body": truncated_response_body,
                    "response_headers": response_headers,
                    "response_time_ms": response_time_ms,
                    "success": success,
                    "error_message": error_message,
                    "retry_count": retry_count,
                    "webhook_metadata": metadata or {},
                }

                webhook_log = WebhookLog(**webhook_log_data)
                db.add(webhook_log)
                await db.commit()
                await db.refresh(webhook_log)

                logger.info(
                    f"Logged webhook attempt for call {call_id}, assistant {assistant_id}, "
                    f"type: {webhook_type}, success: {success}, status: {response_status_code}"
                )

                return webhook_log
        except Exception as e:
            logger.error(f"Error logging webhook attempt: {e}")
        return None

    @staticmethod
    async def send_sms_webhook(
        assistant: Assistant,
        conversation_id: int,
        webhook_type: str,
        from_number: str,
        to_number: str,
        message_body: str,
        direction: str = "incoming",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send an SMS webhook notification.

        Args:
            assistant: The assistant handling the SMS
            conversation_id: The conversation ID
            webhook_type: Type of webhook (sms-received, sms-sent, sms-error)
            from_number: The sender's phone number
            to_number: The recipient's phone number
            message_body: The SMS message content
            direction: Message direction (incoming/outgoing)
            metadata: Additional metadata

        Returns:
            bool: True if webhook was sent successfully, False otherwise
        """
        if not assistant.webhook_url:
            logger.debug(f"No webhook URL configured for assistant {assistant.id}")
            return True  # Not an error if no webhook is configured

        start_time = datetime.now()
        try:
            # Get conversation messages if needed
            messages = []
            if webhook_type == "sms-conversation-complete":
                messages = await WebhookService._get_sms_conversation_messages(
                    conversation_id
                )

            payload = {
                "message": {
                    "type": webhook_type,
                    "phoneNumber": {"number": assistant.phone_number},
                    "customer": {
                        "number": from_number if direction == "incoming" else to_number
                    },
                    "sms": {
                        "from": from_number,
                        "to": to_number,
                        "body": message_body,
                        "direction": direction,
                        "conversationId": conversation_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    },
                    "messages": messages,
                    "metadata": metadata or {},
                }
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    assistant.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                # Calculate response time
                response_time_ms = int(
                    (datetime.now() - start_time).total_seconds() * 1000
                )

                success = response.status_code in [200, 201, 202]
                response_headers = dict(response.headers) if response.headers else None

                # Log the webhook attempt
                await WebhookService.log_webhook_attempt(
                    call_id=conversation_id,  # Using conversation_id as call_id for compatibility
                    assistant_id=assistant.id,
                    webhook_url=assistant.webhook_url,
                    webhook_type=webhook_type,
                    request_payload=payload,
                    response_status_code=response.status_code,
                    response_body=response.text,
                    response_headers=response_headers,
                    response_time_ms=response_time_ms,
                    success=success,
                    metadata={
                        "direction": direction,
                        "from": from_number,
                        "to": to_number,
                    },
                )

                if success:
                    logger.info(
                        f"SMS webhook sent successfully for conversation {conversation_id}"
                    )
                    return True
                else:
                    logger.warning(
                        f"SMS webhook failed with status {response.status_code} for conversation {conversation_id}: {response.text}"
                    )
                    return False

        except Exception as e:
            # Calculate response time even for errors
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Log the failed webhook attempt
            await WebhookService.log_webhook_attempt(
                call_id=conversation_id,
                assistant_id=assistant.id,
                webhook_url=assistant.webhook_url,
                webhook_type=webhook_type,
                request_payload=payload,
                response_time_ms=response_time_ms,
                success=False,
                error_message=str(e),
                metadata={"direction": direction, "from": from_number, "to": to_number},
            )

            logger.error(
                f"Error sending SMS webhook for conversation {conversation_id}: {e}"
            )
            return False

    @staticmethod
    async def _get_sms_conversation_messages(
        conversation_id: int,
    ) -> List[Dict[str, str]]:
        """
        Get SMS conversation messages from chat messages.

        Args:
            conversation_id: The conversation ID

        Returns:
            List of message dictionaries with role and content
        """
        try:
            chat_messages = (
                await ConversationService.get_chat_messages_for_conversation(
                    conversation_id
                )
            )
            messages = []

            for message in chat_messages:
                role = "bot" if message.role == "assistant" else "human"
                messages.append({"role": role, "content": message.content})

            return messages

        except Exception as e:
            logger.error(
                f"Error getting SMS conversation messages for conversation {conversation_id}: {e}"
            )
            return []
