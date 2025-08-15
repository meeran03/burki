"""remove_telephony_config_from_assistants

Revision ID: c66a0d44f5d3
Revises: 499e40be4242
Create Date: 2025-08-13 21:16:28.923514

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c66a0d44f5d3'
down_revision: Union[str, None] = '499e40be4242'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove telephony provider configuration from assistants table."""
    # Remove telephony provider fields from assistants table
    # These are now handled at the organization level only
    op.drop_column('assistants', 'twilio_account_sid')
    op.drop_column('assistants', 'twilio_auth_token')
    op.drop_column('assistants', 'telnyx_api_key')
    op.drop_column('assistants', 'telnyx_connection_id')


def downgrade() -> None:
    """Re-add telephony provider configuration to assistants table."""
    # Re-add the telephony provider fields for rollback
    op.add_column('assistants', sa.Column('twilio_account_sid', sa.String(255), nullable=True))
    op.add_column('assistants', sa.Column('twilio_auth_token', sa.String(255), nullable=True))
    op.add_column('assistants', sa.Column('telnyx_api_key', sa.String(255), nullable=True))
    op.add_column('assistants', sa.Column('telnyx_connection_id', sa.String(255), nullable=True))
