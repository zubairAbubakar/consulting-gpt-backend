"""add search_keywords to technology

Revision ID: 6d10ac74990c
Revises: 96165166a75d
Create Date: 2025-05-14 18:51:28.613568

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6d10ac74990c'
down_revision: Union[str, None] = '96165166a75d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('technology', sa.Column('search_keywords', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('technology', 'search_keywords')
