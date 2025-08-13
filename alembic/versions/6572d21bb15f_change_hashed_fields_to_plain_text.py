"""change_hashed_fields_to_plain_text

Revision ID: 6572d21bb15f
Revises: fb6d9f893224
Create Date: 2025-08-13 13:16:17.358754

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6572d21bb15f'
down_revision: Union[str, Sequence[str], None] = 'fb6d9f893224'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to change hashed fields to plain text."""
    # Drop old indexes
    op.drop_index('ix_candidate_name_hash', table_name='candidate')
    op.drop_index('ix_candidate_email_hash', table_name='candidate')
    
    # Rename columns from hashed to plain text
    op.alter_column('candidate', 'name_hash', new_column_name='name')
    op.alter_column('candidate', 'email_hash', new_column_name='email')  
    op.alter_column('candidate', 'phone_hash', new_column_name='phone')
    
    # Create new indexes on plain text fields
    op.create_index('ix_candidate_name', 'candidate', ['name'])
    op.create_index('ix_candidate_email', 'candidate', ['email'])


def downgrade() -> None:
    """Downgrade schema to restore hashed fields."""
    # Drop new indexes
    op.drop_index('ix_candidate_name', table_name='candidate')
    op.drop_index('ix_candidate_email', table_name='candidate')
    
    # Rename columns back to hashed names
    op.alter_column('candidate', 'name', new_column_name='name_hash')
    op.alter_column('candidate', 'email', new_column_name='email_hash')
    op.alter_column('candidate', 'phone', new_column_name='phone_hash')
    
    # Recreate old indexes
    op.create_index('ix_candidate_name_hash', 'candidate', ['name_hash'])
    op.create_index('ix_candidate_email_hash', 'candidate', ['email_hash'])
