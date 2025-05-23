"""Update Pro plan pricing and limits

Revision ID: d1e2f3g4h5i6
Revises: c8f37d627402
Create Date: 2025-01-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column
from sqlalchemy import String, Integer, Text


# revision identifiers, used by Alembic.
revision: str = 'd1e2f3g4h5i6'
down_revision: Union[str, None] = 'c8f37d627402'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Update Pro plan from unlimited minutes at $20/month to 1000 minutes at $30/month."""
    
    # Create a table object for raw SQL updates
    billing_plans = table('billing_plans',
        column('id', Integer),
        column('name', String),
        column('description', Text),
        column('monthly_minutes', Integer),
        column('price_cents', Integer),
    )
    
    # Update the Pro plan pricing and limits
    op.execute(
        billing_plans.update()
        .where(billing_plans.c.name == 'Pro')
        .values(
            description='1000 minutes for $30/month',
            monthly_minutes=1000,
            price_cents=3000
        )
    )


def downgrade() -> None:
    """Revert Pro plan back to unlimited minutes at $20/month."""
    
    # Create a table object for raw SQL updates
    billing_plans = table('billing_plans',
        column('id', Integer),
        column('name', String),
        column('description', Text),
        column('monthly_minutes', Integer),
        column('price_cents', Integer),
    )
    
    # Revert the Pro plan pricing and limits
    op.execute(
        billing_plans.update()
        .where(billing_plans.c.name == 'Pro')
        .values(
            description='Unlimited minutes for $20/month',
            monthly_minutes=None,
            price_cents=2000
        )
    ) 