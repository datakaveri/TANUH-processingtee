#!/usr/bin/env python3

"""
Loads the active network policy.
"""

import json
from pathlib import Path


POLICY_FILE = Path(__file__).parent / "network_policy.json"


def load_policy():
    """
    Load and return the active network policy.
    """
    with open(POLICY_FILE, "r") as f:
        return json.load(f)