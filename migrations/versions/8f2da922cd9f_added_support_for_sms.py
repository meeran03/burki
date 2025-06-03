"""added support for sms - SAFE VERSION WITH DATA MIGRATION

Revision ID: 8f2da922cd9f
Revises: 801c41697c3b
Create Date: 2025-06-01 22:47:25.322786

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '8f2da922cd9f'
down_revision: Union[str, None] = '801c41697c3b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema with data preservation."""
    
    # STEP 1: Create conversations table
    op.create_table('conversations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('channel_sid', sa.String(length=100), nullable=False),
        sa.Column('conversation_type', sa.String(length=20), nullable=False),
        sa.Column('assistant_id', sa.Integer(), nullable=False),
        sa.Column('to_phone_number', sa.String(length=20), nullable=True),
        sa.Column('customer_phone_number', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('conversation_metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['assistant_id'], ['assistants.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index(op.f('ix_conversations_assistant_id'), 'conversations', ['assistant_id'], unique=False)
    op.create_index(op.f('ix_conversations_channel_sid'), 'conversations', ['channel_sid'], unique=True)
    op.create_index(op.f('ix_conversations_conversation_type'), 'conversations', ['conversation_type'], unique=False)
    op.create_index(op.f('ix_conversations_status'), 'conversations', ['status'], unique=False)
    
    # STEP 2: Migrate data from calls to conversations
    # This preserves all existing call data
    connection = op.get_bind()
    result = connection.execute(
        sa.text("""
            INSERT INTO conversations 
            (id, channel_sid, conversation_type, assistant_id, to_phone_number, 
             customer_phone_number, status, duration, started_at, ended_at, conversation_metadata)
            SELECT 
                id, 
                call_sid as channel_sid,
                'call' as conversation_type,
                assistant_id,
                to_phone_number,
                customer_phone_number,
                status,
                duration,
                started_at,
                ended_at,
                call_meta as conversation_metadata
            FROM calls;
        """)
    )
    
    # Update the sequence to continue from the highest ID
    connection.execute(
        sa.text("SELECT setval('conversations_id_seq', (SELECT COALESCE(MAX(id), 1) FROM conversations));")
    )
    
    # STEP 3: Add new columns to related tables (keeping old ones temporarily)
    op.add_column('assistants', sa.Column('sms_settings', sa.JSON(), nullable=True))
    
    # Add conversation_id columns WITHOUT dropping call_id yet
    op.add_column('chat_messages', sa.Column('conversation_id', sa.Integer(), nullable=True))
    op.add_column('recordings', sa.Column('conversation_id', sa.Integer(), nullable=True))
    op.add_column('transcripts', sa.Column('conversation_id', sa.Integer(), nullable=True))
    op.add_column('usage_records', sa.Column('conversation_id', sa.Integer(), nullable=True))
    op.add_column('usage_records', sa.Column('messages_used', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('webhook_logs', sa.Column('conversation_id', sa.Integer(), nullable=True))
    
    # STEP 4: Copy data from call_id to conversation_id
    connection.execute(sa.text("UPDATE chat_messages SET conversation_id = call_id WHERE call_id IS NOT NULL;"))
    connection.execute(sa.text("UPDATE recordings SET conversation_id = call_id WHERE call_id IS NOT NULL;"))
    connection.execute(sa.text("UPDATE transcripts SET conversation_id = call_id WHERE call_id IS NOT NULL;"))
    connection.execute(sa.text("UPDATE usage_records SET conversation_id = call_id WHERE call_id IS NOT NULL;"))
    connection.execute(sa.text("UPDATE webhook_logs SET conversation_id = call_id WHERE call_id IS NOT NULL;"))
    
    # STEP 5: Add foreign key constraints
    op.create_foreign_key(None, 'chat_messages', 'conversations', ['conversation_id'], ['id'])
    op.create_foreign_key(None, 'recordings', 'conversations', ['conversation_id'], ['id'])
    op.create_foreign_key(None, 'transcripts', 'conversations', ['conversation_id'], ['id'])
    op.create_foreign_key(None, 'usage_records', 'conversations', ['conversation_id'], ['id'])
    op.create_foreign_key(None, 'webhook_logs', 'conversations', ['conversation_id'], ['id'])
    
    # Create new indexes
    op.create_index(op.f('ix_chat_messages_conversation_id'), 'chat_messages', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_recordings_conversation_id'), 'recordings', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_transcripts_conversation_id'), 'transcripts', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_usage_records_conversation_id'), 'usage_records', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_webhook_logs_conversation_id'), 'webhook_logs', ['conversation_id'], unique=False)
    
    # Create new unique constraint for chat_messages
    op.create_index('idx_conversation_message_index', 'chat_messages', ['conversation_id', 'message_index'], unique=True)
    
    # STEP 6: Make conversation_id NOT NULL after data is copied
    op.alter_column('chat_messages', 'conversation_id', nullable=False)
    op.alter_column('recordings', 'conversation_id', nullable=False)
    op.alter_column('transcripts', 'conversation_id', nullable=False)
    op.alter_column('webhook_logs', 'conversation_id', nullable=False)
    # usage_records.conversation_id remains nullable for non-call usage
    
    # STEP 7: Drop old constraints and columns (only after data is safe)
    op.drop_constraint('chat_messages_call_id_fkey', 'chat_messages', type_='foreignkey')
    op.drop_constraint('recordings_call_id_fkey', 'recordings', type_='foreignkey')
    op.drop_constraint('transcripts_call_id_fkey', 'transcripts', type_='foreignkey')
    op.drop_constraint('usage_records_call_id_fkey', 'usage_records', type_='foreignkey')
    op.drop_constraint('webhook_logs_call_id_fkey', 'webhook_logs', type_='foreignkey')
    
    # Drop old indexes
    op.drop_index('idx_call_message_index', table_name='chat_messages')
    op.drop_index('ix_chat_messages_call_id', table_name='chat_messages')
    op.drop_index('ix_recordings_call_id', table_name='recordings')
    op.drop_index('ix_transcripts_call_id', table_name='transcripts')
    op.drop_index('ix_usage_records_call_id', table_name='usage_records')
    op.drop_index('ix_webhook_logs_call_id', table_name='webhook_logs')
    
    # Drop old columns
    op.drop_column('chat_messages', 'call_id')
    op.drop_column('recordings', 'call_id')
    op.drop_column('transcripts', 'call_id')
    op.drop_column('usage_records', 'call_id')
    op.drop_column('webhook_logs', 'call_id')
    
    # STEP 8: Finally, drop the calls table (data already migrated)
    op.drop_index('ix_calls_assistant_id', table_name='calls')
    op.drop_index('ix_calls_status', table_name='calls')
    op.drop_table('calls')


def downgrade() -> None:
    """Downgrade schema with data preservation."""
    
    # STEP 1: Recreate calls table
    op.create_table('calls',
        sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column('call_sid', sa.VARCHAR(length=100), autoincrement=False, nullable=False),
        sa.Column('assistant_id', sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column('to_phone_number', sa.VARCHAR(length=20), autoincrement=False, nullable=False),
        sa.Column('customer_phone_number', sa.VARCHAR(length=20), autoincrement=False, nullable=False),
        sa.Column('status', sa.VARCHAR(length=20), autoincrement=False, nullable=False),
        sa.Column('duration', sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column('started_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
        sa.Column('ended_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
        sa.Column('call_meta', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True),
        sa.ForeignKeyConstraint(['assistant_id'], ['assistants.id'], name='calls_assistant_id_fkey'),
        sa.PrimaryKeyConstraint('id', name='calls_pkey'),
        sa.UniqueConstraint('call_sid', name='calls_call_sid_key')
    )
    op.create_index('ix_calls_status', 'calls', ['status'], unique=False)
    op.create_index('ix_calls_assistant_id', 'calls', ['assistant_id'], unique=False)
    
    # STEP 2: Migrate call conversations back to calls table
    connection = op.get_bind()
    connection.execute(
        sa.text("""
            INSERT INTO calls 
            (id, call_sid, assistant_id, to_phone_number, customer_phone_number, 
             status, duration, started_at, ended_at, call_meta)
            SELECT 
                id,
                channel_sid as call_sid,
                assistant_id,
                to_phone_number,
                customer_phone_number,
                status,
                duration,
                started_at,
                ended_at,
                conversation_metadata as call_meta
            FROM conversations
            WHERE conversation_type = 'call';
        """)
    )
    
    # STEP 3: Add back call_id columns
    op.add_column('webhook_logs', sa.Column('call_id', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('usage_records', sa.Column('call_id', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('transcripts', sa.Column('call_id', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('recordings', sa.Column('call_id', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('chat_messages', sa.Column('call_id', sa.INTEGER(), autoincrement=False, nullable=True))
    
    # STEP 4: Copy data back from conversation_id to call_id (only for calls)
    connection.execute(
        sa.text("""
            UPDATE chat_messages cm 
            SET call_id = cm.conversation_id 
            FROM conversations c 
            WHERE cm.conversation_id = c.id AND c.conversation_type = 'call';
        """)
    )
    connection.execute(
        sa.text("""
            UPDATE recordings r 
            SET call_id = r.conversation_id 
            FROM conversations c 
            WHERE r.conversation_id = c.id AND c.conversation_type = 'call';
        """)
    )
    connection.execute(
        sa.text("""
            UPDATE transcripts t 
            SET call_id = t.conversation_id 
            FROM conversations c 
            WHERE t.conversation_id = c.id AND c.conversation_type = 'call';
        """)
    )
    connection.execute(
        sa.text("""
            UPDATE usage_records ur 
            SET call_id = ur.conversation_id 
            FROM conversations c 
            WHERE ur.conversation_id = c.id AND c.conversation_type = 'call';
        """)
    )
    connection.execute(
        sa.text("""
            UPDATE webhook_logs wl 
            SET call_id = wl.conversation_id 
            FROM conversations c 
            WHERE wl.conversation_id = c.id AND c.conversation_type = 'call';
        """)
    )
    
    # Make call_id NOT NULL for tables that originally required it
    op.alter_column('chat_messages', 'call_id', nullable=False)
    op.alter_column('recordings', 'call_id', nullable=False)
    op.alter_column('transcripts', 'call_id', nullable=False)
    op.alter_column('webhook_logs', 'call_id', nullable=False)
    
    # STEP 5: Recreate constraints and indexes
    op.create_foreign_key('webhook_logs_call_id_fkey', 'webhook_logs', 'calls', ['call_id'], ['id'])
    op.create_foreign_key('usage_records_call_id_fkey', 'usage_records', 'calls', ['call_id'], ['id'])
    op.create_foreign_key('transcripts_call_id_fkey', 'transcripts', 'calls', ['call_id'], ['id'])
    op.create_foreign_key('recordings_call_id_fkey', 'recordings', 'calls', ['call_id'], ['id'])
    op.create_foreign_key('chat_messages_call_id_fkey', 'chat_messages', 'calls', ['call_id'], ['id'])
    
    op.create_index('ix_webhook_logs_call_id', 'webhook_logs', ['call_id'], unique=False)
    op.create_index('ix_usage_records_call_id', 'usage_records', ['call_id'], unique=False)
    op.create_index('ix_transcripts_call_id', 'transcripts', ['call_id'], unique=False)
    op.create_index('ix_recordings_call_id', 'recordings', ['call_id'], unique=False)
    op.create_index('ix_chat_messages_call_id', 'chat_messages', ['call_id'], unique=False)
    op.create_index('idx_call_message_index', 'chat_messages', ['call_id', 'message_index'], unique=True)
    
    # STEP 6: Drop conversation-related constraints and columns
    op.drop_constraint(None, 'webhook_logs', type_='foreignkey')
    op.drop_constraint(None, 'usage_records', type_='foreignkey')
    op.drop_constraint(None, 'transcripts', type_='foreignkey')
    op.drop_constraint(None, 'recordings', type_='foreignkey')
    op.drop_constraint(None, 'chat_messages', type_='foreignkey')
    
    op.drop_index(op.f('ix_webhook_logs_conversation_id'), table_name='webhook_logs')
    op.drop_index(op.f('ix_usage_records_conversation_id'), table_name='usage_records')
    op.drop_index(op.f('ix_transcripts_conversation_id'), table_name='transcripts')
    op.drop_index(op.f('ix_recordings_conversation_id'), table_name='recordings')
    op.drop_index(op.f('ix_chat_messages_conversation_id'), table_name='chat_messages')
    op.drop_index('idx_conversation_message_index', table_name='chat_messages')
    
    op.drop_column('webhook_logs', 'conversation_id')
    op.drop_column('usage_records', 'messages_used')
    op.drop_column('usage_records', 'conversation_id')
    op.drop_column('transcripts', 'conversation_id')
    op.drop_column('recordings', 'conversation_id')
    op.drop_column('chat_messages', 'conversation_id')
    op.drop_column('assistants', 'sms_settings')
    
    # STEP 7: Finally drop conversations table
    op.drop_index(op.f('ix_conversations_status'), table_name='conversations')
    op.drop_index(op.f('ix_conversations_conversation_type'), table_name='conversations')
    op.drop_index(op.f('ix_conversations_channel_sid'), table_name='conversations')
    op.drop_index(op.f('ix_conversations_assistant_id'), table_name='conversations')
    op.drop_table('conversations') 