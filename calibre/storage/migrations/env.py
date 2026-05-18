from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from calibre.storage.models import Base
from calibre.storage.postgres import database_url

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    url = database_url() or config.get_main_option("sqlalchemy.url")
    if not url or url.startswith("driver://"):
        raise RuntimeError(
            "Set CALIBRE_DATABASE_URL or sqlalchemy.url before running storage migrations"
        )
    return url


def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
