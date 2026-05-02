# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Platform helpers: config loading, database and service connections."""

from atelier.config import load_config


def get_config(**overrides):
    """Load config with optional overrides."""
    return load_config(overrides=overrides if overrides else None)


def pg_engine(db_url=None):
    """Create a SQLAlchemy engine for the configured database."""
    from sqlalchemy import create_engine
    url = db_url or get_config().db_url
    return create_engine(url)
