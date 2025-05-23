"""Add billing system tables

Revision ID: c8f37d627402
Revises: 0fddbee709f2
Create Date: 2025-05-23 15:35:30.334311

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8f37d627402'
down_revision: Union[str, None] = '0fddbee709f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Update existing billing_plans table
    op.add_column('billing_plans', sa.Column('monthly_minutes', sa.Integer(), nullable=True))
    op.add_column('billing_plans', sa.Column('price_cents', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('billing_plans', sa.Column('sort_order', sa.Integer(), nullable=False, server_default='0'))
    op.alter_column('billing_plans', 'stripe_price_id', type_=sa.String(length=100))
    
    # Drop old columns
    op.drop_column('billing_plans', 'slug')
    op.drop_column('billing_plans', 'monthly_price')
    op.drop_column('billing_plans', 'minutes_included')
    op.drop_column('billing_plans', 'stripe_product_id')

    # Create billing_accounts table
    op.create_table('billing_accounts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=False),
        sa.Column('plan_id', sa.Integer(), nullable=False),
        sa.Column('stripe_customer_id', sa.String(length=100), nullable=True),
        sa.Column('stripe_subscription_id', sa.String(length=100), nullable=True),
        sa.Column('current_period_minutes_used', sa.Integer(), nullable=False, default=0),
        sa.Column('current_period_start', sa.DateTime(), nullable=False),
        sa.Column('current_period_end', sa.DateTime(), nullable=False),
        sa.Column('auto_topup_enabled', sa.Boolean(), nullable=False, default=False),
        sa.Column('topup_threshold_minutes', sa.Integer(), nullable=False, default=10),
        sa.Column('topup_amount_minutes', sa.Integer(), nullable=False, default=100),
        sa.Column('topup_price_cents', sa.Integer(), nullable=False, default=500),
        sa.Column('status', sa.String(length=20), nullable=False, default='active'),
        sa.Column('is_payment_method_attached', sa.Boolean(), nullable=False, default=False),
        sa.Column('billing_settings', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.ForeignKeyConstraint(['plan_id'], ['billing_plans.id'], ),
        sa.UniqueConstraint('organization_id')
    )
    op.create_index(op.f('ix_billing_accounts_organization_id'), 'billing_accounts', ['organization_id'], unique=False)
    op.create_index(op.f('ix_billing_accounts_plan_id'), 'billing_accounts', ['plan_id'], unique=False)
    op.create_index(op.f('ix_billing_accounts_stripe_customer_id'), 'billing_accounts', ['stripe_customer_id'], unique=False)
    op.create_index(op.f('ix_billing_accounts_stripe_subscription_id'), 'billing_accounts', ['stripe_subscription_id'], unique=False)

    # Update existing usage_records table
    op.add_column('usage_records', sa.Column('billing_account_id', sa.Integer(), nullable=True))
    op.add_column('usage_records', sa.Column('usage_type', sa.String(length=20), nullable=False, server_default='call'))
    op.add_column('usage_records', sa.Column('description', sa.Text(), nullable=True))
    op.add_column('usage_records', sa.Column('billing_period_start', sa.DateTime(), nullable=True))
    op.add_column('usage_records', sa.Column('billing_period_end', sa.DateTime(), nullable=True))
    op.add_column('usage_records', sa.Column('record_metadata', sa.JSON(), nullable=True))
    op.alter_column('usage_records', 'call_id', nullable=True)
    
    # Create foreign key for billing_account_id
    op.create_foreign_key(None, 'usage_records', 'billing_accounts', ['billing_account_id'], ['id'])
    op.create_index(op.f('ix_usage_records_billing_account_id'), 'usage_records', ['billing_account_id'], unique=False)

    # Create billing_transactions table
    op.create_table('billing_transactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('billing_account_id', sa.Integer(), nullable=False),
        sa.Column('transaction_type', sa.String(length=20), nullable=False),
        sa.Column('amount_cents', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, default='usd'),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('stripe_payment_intent_id', sa.String(length=100), nullable=True),
        sa.Column('stripe_invoice_id', sa.String(length=100), nullable=True),
        sa.Column('stripe_charge_id', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, default='pending'),
        sa.Column('minutes_credited', sa.Integer(), nullable=True),
        sa.Column('billing_period_start', sa.DateTime(), nullable=True),
        sa.Column('billing_period_end', sa.DateTime(), nullable=True),
        sa.Column('transaction_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['billing_account_id'], ['billing_accounts.id'], )
    )
    op.create_index(op.f('ix_billing_transactions_billing_account_id'), 'billing_transactions', ['billing_account_id'], unique=False)
    op.create_index(op.f('ix_billing_transactions_stripe_charge_id'), 'billing_transactions', ['stripe_charge_id'], unique=False)
    op.create_index(op.f('ix_billing_transactions_stripe_invoice_id'), 'billing_transactions', ['stripe_invoice_id'], unique=False)
    op.create_index(op.f('ix_billing_transactions_stripe_payment_intent_id'), 'billing_transactions', ['stripe_payment_intent_id'], unique=False)
    op.create_index(op.f('ix_billing_transactions_transaction_type'), 'billing_transactions', ['transaction_type'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop new tables
    op.drop_table('billing_transactions')
    op.drop_table('billing_accounts')
    
    # Revert usage_records changes
    op.drop_index(op.f('ix_usage_records_billing_account_id'), table_name='usage_records')
    op.drop_column('usage_records', 'record_metadata')
    op.drop_column('usage_records', 'billing_period_end')
    op.drop_column('usage_records', 'billing_period_start')
    op.drop_column('usage_records', 'description')
    op.drop_column('usage_records', 'usage_type')
    op.drop_column('usage_records', 'billing_account_id')
    op.alter_column('usage_records', 'call_id', nullable=False)
    
    # Revert billing_plans changes
    op.add_column('billing_plans', sa.Column('stripe_product_id', sa.String(length=255), nullable=True))
    op.add_column('billing_plans', sa.Column('minutes_included', sa.Integer(), nullable=True))
    op.add_column('billing_plans', sa.Column('monthly_price', sa.Float(), nullable=True))
    op.add_column('billing_plans', sa.Column('slug', sa.String(length=100), nullable=True))
    op.drop_column('billing_plans', 'sort_order')
    op.drop_column('billing_plans', 'price_cents')
    op.drop_column('billing_plans', 'monthly_minutes')
    op.alter_column('billing_plans', 'stripe_price_id', type_=sa.String(length=255))
