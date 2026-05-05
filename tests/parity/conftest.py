"""Pytest config for the parity test suite.

Registers a `release` Hypothesis profile that runs 1000 examples per test
(used by CI on tag pushes). Default profile keeps it at 100.
"""
from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "release",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Pytest-hypothesis auto-loads conftest.py and applies whichever profile
# is selected via --hypothesis-profile=NAME on the command line.
