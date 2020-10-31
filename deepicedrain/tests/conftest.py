"""
This module contains shared fixtures, steps, and hooks.
"""
import pytest


@pytest.fixture
def context():
    """
    An empty context class, used for passing arbitrary objects between tests.
    """

    class Context:
        pass

    return Context()
