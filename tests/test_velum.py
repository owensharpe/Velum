"""
Title: test_velum.py
Author: Owen Sharpe
Description: Setting up a basic test such that make test works.
"""

import velum


def test_version():
    assert velum.__version__ == "0.1.0"
