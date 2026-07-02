#!/usr/bin/env python3
"""Download objects from GCS using the instance OAuth2 token."""

import urllib.parse
import requests
from Fetch_data.secrets import get_oauth_token


def download_gcs_object(bucket: str, object_path: str, dest_path: str) -> None:
    """
    Download a GCS object to a local file using the instance OAuth2 token.

    Args:
        bucket: GCS bucket name, e.g. 'tanuh-benchmark-datasets'
        object_path: Object name within bucket, e.g. 'datasets/dataset_1.enc'
        dest_path: Local file path to write downloaded bytes to
    """
    token = get_oauth_token()
    encoded_object = urllib.parse.quote(object_path, safe="")
    url = f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{encoded_object}?alt=media"

    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
        stream=True,
    )
    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
