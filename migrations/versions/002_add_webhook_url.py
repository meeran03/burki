"""Add webhook_url to Assistant model

Revision ID: 002_add_webhook_url
Revises: 001_initial_migration
Create Date: 2023-05-18 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002_add_webhook_url'
down_revision = '001_initial_migration'
branch_labels = None
depends_on = None


def upgrade():
    # Add webhook_url column to assistants table
    op.add_column('assistants', sa.Column('webhook_url', sa.String(length=255), nullable=True))


def downgrade():
    # Remove webhook_url column from assistants table
    op.drop_column('assistants', 'webhook_url') 