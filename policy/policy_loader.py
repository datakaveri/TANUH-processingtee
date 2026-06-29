#!/usr/bin/env python3

"""
Loads the active network policy.
"""

import json
from pathlib import Path


POLICIES = Path(__file__).parent / "network_policy.json"


def get_policy():

    with open(POLICIES) as f:
        return json.load(f)