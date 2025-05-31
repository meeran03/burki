"""
S3 service for handling audio file storage in AWS S3.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from io import BytesIO
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

logger = logging.getLogger(__name__)


class S3Service:
    """
    Service for handling AWS S3 operations for audio recordings.
    """

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ):
        """
        Initialize S3 service.

        Args:
            aws_access_key_id: AWS access key ID (optional, will use env var if not provided)
            aws_secret_access_key: AWS secret access key (optional, will use env var if not provided)
            aws_region: AWS region (optional, will use env var if not provided)
            bucket_name: S3 bucket name (optional, will use env var if not provided)
        """
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET_NAME")

        if not all([self.aws_access_key_id, self.aws_secret_access_key, self.bucket_name]):
            logger.error("Missing required AWS S3 configuration")
            raise ValueError("AWS S3 credentials and bucket name are required")

        # Configure S3 client with retry and timeout settings
        config = Config(
            region_name=self.aws_region,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            },
            max_pool_connections=50,
            connect_timeout=60,
            read_timeout=60
        )

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region,
            config=config
        )

        logger.info(f"S3Service initialized for bucket: {self.bucket_name} in region: {self.aws_region}")

    async def upload_audio_file(
        self,
        audio_data: bytes,
        call_sid: str,
        recording_type: str,
        format: str = "mp3",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """
        Upload audio file to S3.

        Args:
            audio_data: Audio data as bytes
            call_sid: Call SID for organizing files
            recording_type: Type of recording (user, assistant, mixed)
            format: Audio format (mp3, wav)
            metadata: Additional metadata to store with the file

        Returns:
            Tuple[str, str]: S3 key and public URL
        """
        try:
            # Generate S3 key with organized structure
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            s3_key = f"recordings/{call_sid}/{recording_type}_{timestamp}.{format}"

            # Prepare metadata
            upload_metadata = {
                "call_sid": call_sid,
                "recording_type": recording_type,
                "format": format,
                "uploaded_at": datetime.utcnow().isoformat(),
                "content_type": f"audio/{format}",
            }
            if metadata:
                upload_metadata.update(metadata)

            # Determine content type
            content_type = "audio/mpeg" if format == "mp3" else "audio/wav"

            # Upload to S3 in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._upload_file_sync,
                audio_data,
                s3_key,
                content_type,
                upload_metadata,
            )

            # Generate public URL
            s3_url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_key}"

            logger.info(f"Successfully uploaded audio file to S3: {s3_key}")
            return s3_key, s3_url

        except Exception as e:
            logger.error(f"Error uploading audio file to S3: {e}")
            raise

    def _upload_file_sync(
        self,
        audio_data: bytes,
        s3_key: str,
        content_type: str,
        metadata: Dict[str, str],
    ) -> None:
        """
        Synchronous upload method to be run in executor.
        """
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=audio_data,
            ContentType=content_type,
            Metadata=metadata,
            ServerSideEncryption='AES256',  # Enable server-side encryption
        )

    async def download_audio_file(self, s3_key: str) -> Optional[bytes]:
        """
        Download audio file from S3.

        Args:
            s3_key: S3 key of the file to download

        Returns:
            Optional[bytes]: Audio data or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._download_file_sync,
                s3_key,
            )
            
            if response:
                logger.info(f"Successfully downloaded audio file from S3: {s3_key}")
                return response['Body'].read()
            return None

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Audio file not found in S3: {s3_key}")
                return None
            else:
                logger.error(f"Error downloading audio file from S3: {e}")
                raise
        except Exception as e:
            logger.error(f"Error downloading audio file from S3: {e}")
            raise

    def _download_file_sync(self, s3_key: str) -> Optional[Dict]:
        """
        Synchronous download method to be run in executor.
        """
        try:
            return self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    async def generate_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        method: str = "get_object",
    ) -> Optional[str]:
        """
        Generate a presigned URL for accessing S3 objects.

        Args:
            s3_key: S3 key of the file
            expiration: URL expiration time in seconds (default: 1 hour)
            method: HTTP method (get_object, put_object)

        Returns:
            Optional[str]: Presigned URL or None if error
        """
        try:
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(
                None,
                self._generate_presigned_url_sync,
                s3_key,
                expiration,
                method,
            )
            
            logger.info(f"Generated presigned URL for S3 key: {s3_key}")
            return url

        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def _generate_presigned_url_sync(
        self, s3_key: str, expiration: int, method: str
    ) -> str:
        """
        Synchronous presigned URL generation method to be run in executor.
        """
        return self.s3_client.generate_presigned_url(
            method,
            Params={'Bucket': self.bucket_name, 'Key': s3_key},
            ExpiresIn=expiration
        )

    async def delete_audio_file(self, s3_key: str) -> bool:
        """
        Delete audio file from S3.

        Args:
            s3_key: S3 key of the file to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._delete_file_sync,
                s3_key,
            )
            
            logger.info(f"Successfully deleted audio file from S3: {s3_key}")
            return True

        except Exception as e:
            logger.error(f"Error deleting audio file from S3: {e}")
            return False

    def _delete_file_sync(self, s3_key: str) -> None:
        """
        Synchronous delete method to be run in executor.
        """
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)

    async def list_call_recordings(self, call_sid: str) -> list:
        """
        List all recordings for a specific call.

        Args:
            call_sid: Call SID to filter recordings

        Returns:
            list: List of S3 objects for the call
        """
        try:
            prefix = f"recordings/{call_sid}/"
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._list_objects_sync,
                prefix,
            )
            
            objects = response.get('Contents', [])
            logger.info(f"Found {len(objects)} recordings for call {call_sid}")
            return objects

        except Exception as e:
            logger.error(f"Error listing recordings for call {call_sid}: {e}")
            return []

    def _list_objects_sync(self, prefix: str) -> Dict:
        """
        Synchronous list objects method to be run in executor.
        """
        return self.s3_client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix
        )

    async def get_file_metadata(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an S3 object.

        Args:
            s3_key: S3 key of the file

        Returns:
            Optional[Dict[str, Any]]: File metadata or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._get_metadata_sync,
                s3_key,
            )
            
            if response:
                metadata = {
                    'size': response.get('ContentLength', 0),
                    'last_modified': response.get('LastModified'),
                    'content_type': response.get('ContentType'),
                    'metadata': response.get('Metadata', {}),
                    'etag': response.get('ETag', '').strip('"'),
                }
                return metadata
            return None

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"File not found in S3: {s3_key}")
                return None
            else:
                logger.error(f"Error getting file metadata from S3: {e}")
                raise
        except Exception as e:
            logger.error(f"Error getting file metadata from S3: {e}")
            raise

    def _get_metadata_sync(self, s3_key: str) -> Optional[Dict]:
        """
        Synchronous metadata retrieval method to be run in executor.
        """
        try:
            return self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def test_connection(self) -> bool:
        """
        Test S3 connection and bucket access.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 connection test successful for bucket: {self.bucket_name}")
            return True
        except Exception as e:
            logger.error(f"S3 connection test failed: {e}")
            return False

    @classmethod
    def create_default_instance(cls) -> "S3Service":
        """
        Create S3Service instance with default environment configuration.

        Returns:
            S3Service: Configured S3Service instance
        """
        return cls()

    async def upload_document(
        self,
        document_data: bytes,
        assistant_id: int,
        original_filename: str,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """
        Upload document file to S3 for RAG functionality.

        Args:
            document_data: Document data as bytes
            assistant_id: Assistant ID for organizing documents
            original_filename: Original filename of the document
            content_type: MIME type of the document
            metadata: Additional metadata to store with the document

        Returns:
            Tuple[str, str]: S3 key and public URL
        """
        try:
            # Generate S3 key with organized structure
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Clean filename for S3
            safe_filename = self._sanitize_filename(original_filename)
            s3_key = f"documents/assistant_{assistant_id}/{timestamp}_{safe_filename}"

            # Prepare metadata
            upload_metadata = {
                "assistant_id": str(assistant_id),
                "original_filename": original_filename,
                "uploaded_at": datetime.utcnow().isoformat(),
                "content_type": content_type,
                "document_type": "knowledge_base",
            }
            if metadata:
                upload_metadata.update(metadata)

            # Upload to S3 in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._upload_file_sync,
                document_data,
                s3_key,
                content_type,
                upload_metadata,
            )

            # Generate public URL
            s3_url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_key}"

            logger.info(f"Successfully uploaded document to S3: {s3_key}")
            return s3_key, s3_url

        except Exception as e:
            logger.error(f"Error uploading document to S3: {e}")
            raise

    async def download_document(self, s3_key: str) -> Optional[bytes]:
        """
        Download document from S3.

        Args:
            s3_key: S3 key of the document to download

        Returns:
            Optional[bytes]: Document data or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._download_file_sync,
                s3_key,
            )
            
            if response:
                logger.info(f"Successfully downloaded document from S3: {s3_key}")
                return response['Body'].read()
            return None

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Document not found in S3: {s3_key}")
                return None
            else:
                logger.error(f"Error downloading document from S3: {e}")
                raise
        except Exception as e:
            logger.error(f"Error downloading document from S3: {e}")
            raise

    async def delete_document(self, s3_key: str) -> bool:
        """
        Delete document from S3.

        Args:
            s3_key: S3 key of the document to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._delete_file_sync,
                s3_key,
            )
            
            logger.info(f"Successfully deleted document from S3: {s3_key}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document from S3: {e}")
            return False

    async def list_assistant_documents(self, assistant_id: int) -> list:
        """
        List all documents for a specific assistant.

        Args:
            assistant_id: Assistant ID to filter documents

        Returns:
            list: List of S3 objects for the assistant's documents
        """
        try:
            prefix = f"documents/assistant_{assistant_id}/"
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._list_objects_sync,
                prefix,
            )
            
            objects = response.get('Contents', [])
            logger.info(f"Found {len(objects)} documents for assistant {assistant_id}")
            return objects

        except Exception as e:
            logger.error(f"Error listing documents for assistant {assistant_id}: {e}")
            return []

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for S3 storage.

        Args:
            filename: Original filename

        Returns:
            str: Sanitized filename safe for S3
        """
        import re
        # Remove unsafe characters and replace with underscores
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        # Remove multiple consecutive underscores
        safe_filename = re.sub(r'_+', '_', safe_filename)
        # Ensure filename is not too long
        if len(safe_filename) > 100:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:90] + ext
        return safe_filename

    async def copy_document(self, source_s3_key: str, destination_s3_key: str) -> bool:
        """
        Copy a document within S3.

        Args:
            source_s3_key: Source S3 key
            destination_s3_key: Destination S3 key

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._copy_file_sync,
                source_s3_key,
                destination_s3_key,
            )
            
            logger.info(f"Successfully copied document from {source_s3_key} to {destination_s3_key}")
            return True

        except Exception as e:
            logger.error(f"Error copying document from {source_s3_key} to {destination_s3_key}: {e}")
            return False

    def _copy_file_sync(self, source_s3_key: str, destination_s3_key: str) -> None:
        """
        Synchronous copy method to be run in executor.
        """
        copy_source = {'Bucket': self.bucket_name, 'Key': source_s3_key}
        self.s3_client.copy_object(
            CopySource=copy_source,
            Bucket=self.bucket_name,
            Key=destination_s3_key,
            ServerSideEncryption='AES256'
        )

    async def get_document_content_type(self, s3_key: str) -> Optional[str]:
        """
        Get the content type of a document in S3.

        Args:
            s3_key: S3 key of the document

        Returns:
            Optional[str]: Content type or None if not found
        """
        metadata = await self.get_file_metadata(s3_key)
        if metadata:
            return metadata.get('content_type')
        return None

    async def check_document_exists(self, s3_key: str) -> bool:
        """
        Check if a document exists in S3.

        Args:
            s3_key: S3 key of the document

        Returns:
            bool: True if document exists, False otherwise
        """
        try:
            metadata = await self.get_file_metadata(s3_key)
            return metadata is not None
        except Exception:
            return False 