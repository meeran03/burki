"""add_transfer_call_message_field

Revision ID: b51d20b49e89
Revises: b01a56100ecc
Create Date: 2025-05-22 19:36:29.895223

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b51d20b49e89'
down_revision: Union[str, None] = 'b01a56100ecc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
