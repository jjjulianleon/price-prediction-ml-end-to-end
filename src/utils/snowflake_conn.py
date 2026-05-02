"""Snowflake connection helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from src.utils.config import Settings, get_settings, get_snowflake_connection_params

try:
    import snowflake.connector
except ImportError:  # pragma: no cover - exercised indirectly in environments without the dependency
    snowflake = None


def create_connection(settings: Settings | None = None):
    """Create a Snowflake connection using environment-backed settings."""
    if snowflake is None:
        raise ImportError(
            "snowflake-connector-python is not installed. "
            "Install project requirements before connecting to Snowflake."
        )

    effective_settings = settings or get_settings()
    params = get_snowflake_connection_params(effective_settings)
    return snowflake.connector.connect(**params)


@contextmanager
def get_cursor(settings: Settings | None = None) -> Iterator:
    connection = create_connection(settings=settings)
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        close_quietly(cursor)
        close_quietly(connection)


def close_quietly(resource) -> None:
    """Close cursor/connection objects without masking prior exceptions."""
    if resource is None:
        return

    try:
        resource.close()
    except Exception:
        return
