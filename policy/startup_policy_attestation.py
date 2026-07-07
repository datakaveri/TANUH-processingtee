#!/usr/bin/env python3

import subprocess
import json
import os
import base64

from policy.policy_hash import get_policy_hash
from policy.policy_loader import load_policy


SOCKET_PATH = "/run/container_launcher/teeserver.sock"


def decode_jwt(token):

    payload = token.split(".")[1]

    payload += "=" * (-len(payload) % 4)

    return json.loads(
        base64.urlsafe_b64decode(payload)
    )


def main():

    print("\n================================================")
    print(" POLICY STARTUP ATTESTATION")
    print("================================================")

    # load policy first
    policy = load_policy()

    print("\nENFORCED NETWORK RULES:\n")

    for idx, rule in enumerate(policy["rules"], start=1):

        print(f"Rule {idx}")

        print(" Destination:",
              rule.get("destination",
              rule.get("destinations")))

        print(" Protocol:",
              rule.get("protocol"))

        print(" Ports:",
              rule.get("ports"))

        print(" Download:",
              rule.get("download", "N/A"))

        print(" Upload:",
              rule.get("upload", "N/A"))

        print(" Description:",
              rule.get("description"))

        print("--------------------------------")


    policy_hash = get_policy_hash()

    print("\nPolicy SHA256:")
    print(policy_hash)


    request = {
        "audience": os.getenv(
            "RATLS_AUDIENCE",
            "ratls-buffer-tee"
        ),
        "token_type": "OIDC",
        "nonces": [
            policy_hash
        ]
    }


    print("\nRequesting startup JWT...\n")


    result = subprocess.run(
        [
            "curl",
            "--silent",
            "--unix-socket",
            SOCKET_PATH,
            "-X",
            "POST",
            "http://localhost/v1/token",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(request)
        ],
        capture_output=True,
        text=True,
        check=True
    )


    token = result.stdout.strip()


    print("\nJWT:")
    print(token)


    decoded = decode_jwt(token)


    print("\nDecoded JWT:")
    print(
        json.dumps(
            decoded,
            indent=4
        )
    )


    print("\nEAT NONCE:")
    print(
        decoded.get("eat_nonce")
    )


    if policy_hash in str(decoded.get("eat_nonce")):
        print(
            "\nPASS: Policy hash verified inside JWT"
        )
    else:
        print(
            "\nFAIL: Policy hash missing"
        )


    print("\n================================================")
    print(" Startup Policy Attestation Complete")
    print("================================================\n")



if __name__ == "__main__":
    main()