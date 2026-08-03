"""Data directory: terrain JSON files for environments."""

import importlib.resources
import os


def data_path(filename: str) -> str:
    """Return absolute path to a data file in this package."""
    return os.path.join(os.path.dirname(__file__), filename)
