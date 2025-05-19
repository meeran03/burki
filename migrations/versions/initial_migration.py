"""Initial migration for VAPI Clone

Revision ID: 001_initial_migration
Revises: 
Create Date: 2025-05-17 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial_migration'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create assistants table
    op.create_table(
        'assistants',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('phone_number', sa.String(length=20), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('openai_api_key', sa.String(length=255), nullable=True),
        sa.Column('custom_llm_url', sa.String(length=255), nullable=True),
        sa.Column('deepgram_api_key', sa.String(length=255), nullable=True),
        sa.Column('elevenlabs_api_key', sa.String(length=255), nullable=True),
        sa.Column('twilio_account_sid', sa.String(length=255), nullable=True),
        sa.Column('twilio_auth_token', sa.String(length=255), nullable=True),
        sa.Column('system_prompt', sa.Text(), nullable=True),
        sa.Column('elevenlabs_voice_id', sa.String(length=100), nullable=True),
        sa.Column('openai_model', sa.String(length=100), nullable=True),
        sa.Column('openai_temperature', sa.Float(), nullable=True),
        sa.Column('openai_max_tokens', sa.Integer(), nullable=True),
        sa.Column('silence_min_duration_ms', sa.Integer(), nullable=True),
        sa.Column('energy_threshold', sa.Integer(), nullable=True),
        sa.Column('wait_after_speech_ms', sa.Integer(), nullable=True),
        sa.Column('no_punctuation_wait_ms', sa.Integer(), nullable=True),
        sa.Column('voice_seconds_threshold', sa.Integer(), nullable=True),
        sa.Column('word_count_threshold', sa.Integer(), nullable=True),
        sa.Column('end_call_message', sa.String(length=255), nullable=True),
        sa.Column('max_idle_messages', sa.Integer(), nullable=True),
        sa.Column('idle_timeout', sa.Integer(), nullable=True),
        sa.Column('custom_settings', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('phone_number')
    )
    
    # Create calls table
    op.create_table(
        'calls',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('call_sid', sa.String(length=100), nullable=False),
        sa.Column('assistant_id', sa.Integer(), nullable=False),
        sa.Column('to_phone_number', sa.String(length=20), nullable=False),
        sa.Column('customer_phone_number', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['assistant_id'], ['assistants.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('call_sid')
    )
    
    # Create recordings table
    op.create_table(
        'recordings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('call_id', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.String(length=255), nullable=False),
        sa.Column('duration', sa.Float(), nullable=True),
        sa.Column('format', sa.String(length=20), nullable=True),
        sa.Column('recording_type', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['call_id'], ['calls.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create transcripts table
    op.create_table(
        'transcripts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('call_id', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('is_final', sa.Boolean(), nullable=True),
        sa.Column('segment_start', sa.Float(), nullable=True),
        sa.Column('segment_end', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('speaker', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['call_id'], ['calls.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index(op.f('ix_calls_assistant_id'), 'calls', ['assistant_id'], unique=False)
    op.create_index(op.f('ix_calls_status'), 'calls', ['status'], unique=False)
    op.create_index(op.f('ix_recordings_call_id'), 'recordings', ['call_id'], unique=False)
    op.create_index(op.f('ix_transcripts_call_id'), 'transcripts', ['call_id'], unique=False)
    op.create_index(op.f('ix_transcripts_speaker'), 'transcripts', ['speaker'], unique=False)


def downgrade():
    # Drop tables in reverse order of creation
    op.drop_index(op.f('ix_transcripts_speaker'), table_name='transcripts')
    op.drop_index(op.f('ix_transcripts_call_id'), table_name='transcripts')
    op.drop_index(op.f('ix_recordings_call_id'), table_name='recordings')
    op.drop_index(op.f('ix_calls_status'), table_name='calls')
    op.drop_index(op.f('ix_calls_assistant_id'), table_name='calls')
    op.drop_table('transcripts')
    op.drop_table('recordings')
    op.drop_table('calls')
    op.drop_table('assistants') 