#!/usr/bin/env python3
"""
policy_hash.py

Computes a deterministic SHA256 hash of the active network policy.

This hash is later embedded into the Confidential Computing attestation
(EAT nonce) so that the verifier can prove which policy was enforced
when the workload executed.
"""

import hashlib
import json
from pathlib import Path


POLICY_FILE = Path(__file__).parent / "network_policy.json"


def load_policy():
    """
    Load the network policy JSON.

    Returns:
        dict
    """
    with open(POLICY_FILE, "r") as f:
        return json.load(f)


def canonical_policy_json(policy: dict) -> str:
    """
    Convert the policy into a deterministic JSON string.

    We remove formatting differences by:
      - sorting keys
      - removing whitespace

    Returns:
        str
    """
    return json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":")
    )


def calculate_policy_hash() -> str:
    """
    Compute SHA256 hash of the policy.

    Returns:
        Hex string.
    """
    policy = load_policy()
    canonical = canonical_policy_json(policy)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_policy_hash() -> str:
    """
    Returns the SHA256 hash of the active network policy.
    """
    return calculate_policy_hash()


if __name__ == "__main__":
    print("Policy SHA256")
    print(calculate_policy_hash())
