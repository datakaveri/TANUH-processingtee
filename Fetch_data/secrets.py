#!/usr/bin/env python3
"""GCP metadata server helpers: OAuth2 token + Secret Manager access."""

import base64
import time
import requests

_METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"
_METADATA_HDR = {"Metadata-Flavor": "Google"}


def get_oauth_token() -> str:
    """Get the OAuth2 access token for the attached Service Account from the metadata server."""
    resp = requests.get(
        f"{_METADATA_ROOT}/instance/service-accounts/default/token",
        headers=_METADATA_HDR,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_oidc_token(audience: str) -> str:
    """Get the OIDC identity token (hardware-attested) from the metadata server."""
    resp = requests.get(
        f"{_METADATA_ROOT}/instance/service-accounts/default/identity"
        f"?audience={audience}&format=full",
        headers=_METADATA_HDR,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.text.strip()


def fetch_secret(project_id: str, secret_id: str, version: str = "latest") -> bytes:
    """
    Fetch a secret's payload bytes from GCP Secret Manager.

    Uses the instance OAuth2 token (from get_oauth_token) — the metadata server
    transparently enforces WIF attribute conditions before issuing this token.

    Returns raw bytes (for Fernet keys: 44-byte URL-safe base64-encoded key).
    """
    token = get_oauth_token()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    url = f"https://secretmanager.googleapis.com/v1/{name}:access"
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return base64.b64decode(resp.json()["payload"]["data"])
        except requests.exceptions.Timeout:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
