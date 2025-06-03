"""
Migration script to convert Call table to Conversation table.
This script handles the transition from call-specific model to a generic conversation model.

Run this script to:
1. Create the new conversations table
2. Migrate existing call data
3. Update foreign key references
4. Add backward compatibility

Note: This migration preserves all existing data and maintains backward compatibility.
"""

import logging
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, Boolean, Float, JSON, ForeignKey, Index
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/diwaar")


def create_conversations_table(engine):
    """Create the new conversations table."""
    with engine.connect() as conn:
        # Check if conversations table already exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'conversations'
            );
        """)).scalar()
        
        if result:
            logger.info("Conversations table already exists")
            return False
        
        logger.info("Creating conversations table...")
        
        # Create conversations table with same structure as calls but renamed fields
        conn.execute(text("""
            CREATE TABLE conversations (
                id SERIAL PRIMARY KEY,
                channel_sid VARCHAR(100) UNIQUE NOT NULL,
                conversation_type VARCHAR(20) NOT NULL,
                assistant_id INTEGER NOT NULL REFERENCES assistants(id),
                to_phone_number VARCHAR(20),
                customer_phone_number VARCHAR(20) NOT NULL,
                status VARCHAR(20) NOT NULL,
                duration INTEGER,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                conversation_metadata JSON,
                CONSTRAINT fk_assistant FOREIGN KEY (assistant_id) REFERENCES assistants(id)
            );
            
            -- Create indexes
            CREATE INDEX idx_conversations_channel_sid ON conversations(channel_sid);
            CREATE INDEX idx_conversations_type ON conversations(conversation_type);
            CREATE INDEX idx_conversations_assistant_id ON conversations(assistant_id);
            CREATE INDEX idx_conversations_status ON conversations(status);
        """))
        
        conn.commit()
        logger.info("Conversations table created successfully")
        return True


def migrate_calls_to_conversations(engine):
    """Migrate existing call records to conversations table."""
    with engine.connect() as conn:
        # Check if calls table exists and has data
        calls_exist = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'calls'
            );
        """)).scalar()
        
        if not calls_exist:
            logger.info("No calls table found, skipping migration")
            return
        
        # Count existing calls
        call_count = conn.execute(text("SELECT COUNT(*) FROM calls")).scalar()
        logger.info(f"Found {call_count} calls to migrate")
        
        if call_count == 0:
            return
        
        # Migrate calls to conversations
        logger.info("Migrating calls to conversations...")
        conn.execute(text("""
            INSERT INTO conversations (
                id,
                channel_sid,
                conversation_type,
                assistant_id,
                to_phone_number,
                customer_phone_number,
                status,
                duration,
                started_at,
                ended_at,
                conversation_metadata
            )
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
            FROM calls
            ON CONFLICT (channel_sid) DO NOTHING;
        """))
        
        # Update sequence to continue from the highest ID
        conn.execute(text("""
            SELECT setval('conversations_id_seq', (SELECT MAX(id) FROM conversations));
        """))
        
        conn.commit()
        logger.info(f"Successfully migrated {call_count} calls to conversations")


def update_foreign_key_references(engine):
    """Update foreign key references from call_id to conversation_id."""
    tables_to_update = [
        ('recordings', 'call_id', 'conversation_id'),
        ('transcripts', 'call_id', 'conversation_id'),
        ('chat_messages', 'call_id', 'conversation_id'),
        ('webhook_logs', 'call_id', 'conversation_id'),
        ('usage_records', 'call_id', 'conversation_id'),
    ]
    
    with engine.connect() as conn:
        for table_name, old_column, new_column in tables_to_update:
            # Check if table exists
            table_exists = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table_name}'
                );
            """)).scalar()
            
            if not table_exists:
                logger.info(f"Table {table_name} does not exist, skipping")
                continue
            
            # Check if column already exists
            column_exists = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table_name}'
                    AND column_name = '{new_column}'
                );
            """)).scalar()
            
            if column_exists:
                logger.info(f"Column {new_column} already exists in {table_name}")
                continue
            
            logger.info(f"Adding {new_column} to {table_name}...")
            
            # Add new column
            conn.execute(text(f"""
                ALTER TABLE {table_name} 
                ADD COLUMN IF NOT EXISTS {new_column} INTEGER REFERENCES conversations(id);
            """))
            
            # Copy data from old column to new column
            conn.execute(text(f"""
                UPDATE {table_name} 
                SET {new_column} = {old_column}
                WHERE {old_column} IS NOT NULL;
            """))
            
            # Add index on new column
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_{new_column} 
                ON {table_name}({new_column});
            """))
            
            logger.info(f"Successfully updated {table_name}")
        
        conn.commit()


def update_unique_constraints(engine):
    """Update unique constraints that reference call_id."""
    with engine.connect() as conn:
        # Update chat_messages unique constraint
        try:
            # Drop old constraint if exists
            conn.execute(text("""
                ALTER TABLE chat_messages 
                DROP CONSTRAINT IF EXISTS idx_call_message_index;
            """))
            
            # Create new constraint
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_message_index 
                ON chat_messages(conversation_id, message_index);
            """))
            
            logger.info("Updated chat_messages unique constraint")
        except Exception as e:
            logger.warning(f"Could not update chat_messages constraint: {e}")
        
        conn.commit()


def main():
    """Run the migration."""
    logger.info("Starting Call to Conversation migration...")
    
    # Create engine
    engine = create_engine(DATABASE_URL)
    
    try:
        # Step 1: Create conversations table
        create_conversations_table(engine)
        
        # Step 2: Migrate existing calls
        migrate_calls_to_conversations(engine)
        
        # Step 3: Update foreign key references
        update_foreign_key_references(engine)
        
        # Step 4: Update unique constraints
        update_unique_constraints(engine)
        
        logger.info("Migration completed successfully!")
        logger.info("""
        Next steps:
        1. Update your application code to use Conversation model
        2. Test thoroughly to ensure backward compatibility works
        3. Once stable, you can optionally drop the old call_id columns
        4. Finally, you can drop the calls table if no longer needed
        """)
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    main() 