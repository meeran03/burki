"""add_llm_provider_support_with_defaults

Revision ID: 0fddbee709f2
Revises: c36d90bbe8cf
Create Date: 2024-12-19 18:31:45.751844

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0fddbee709f2'
down_revision = 'c36d90bbe8cf'
branch_labels = None
depends_on = None


def upgrade():
    # Step 1: Add llm_provider column as nullable with default
    op.add_column('assistants', sa.Column('llm_provider', sa.String(length=50), nullable=True))
    
    # Step 2: Add llm_provider_config column as nullable 
    op.add_column('assistants', sa.Column('llm_provider_config', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # Step 3: Update existing records to set default values
    # For existing assistants, set provider based on what they currently have
    connection = op.get_bind()
    
    # Set provider to 'custom' if they have custom_llm_url, otherwise 'openai'
    connection.execute(sa.text("""
        UPDATE assistants 
        SET llm_provider = CASE 
            WHEN custom_llm_url IS NOT NULL AND custom_llm_url != '' THEN 'custom'
            ELSE 'openai'
        END
        WHERE llm_provider IS NULL
    """))
    
    # Set llm_provider_config based on existing configuration
    connection.execute(sa.text("""
        UPDATE assistants 
        SET llm_provider_config = CASE 
            WHEN custom_llm_url IS NOT NULL AND custom_llm_url != '' THEN 
                json_build_object('base_url', custom_llm_url)
            WHEN openai_api_key IS NOT NULL AND openai_api_key != '' THEN 
                json_build_object('api_key', openai_api_key, 'model', 'gpt-4o-mini')
            ELSE 
                json_build_object('model', 'gpt-4o-mini')
        END
        WHERE llm_provider_config IS NULL
    """))
    
    # Step 4: Now make llm_provider NOT NULL since all records have values
    op.alter_column('assistants', 'llm_provider', nullable=False)


def downgrade():
    # Remove the new columns
    op.drop_column('assistants', 'llm_provider_config')
    op.drop_column('assistants', 'llm_provider')
