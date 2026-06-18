#!/usr/bin/env python3
"""Download and upload objects to GCS using the instance OAuth2 token."""

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


def download_gcs_object_bytes(bucket: str, object_path: str) -> bytes:
    """
    Download a GCS object and return its content as bytes.
    Raises requests.HTTPError on failure (including 404 if object doesn't exist).
    """
    token = get_oauth_token()
    encoded_object = urllib.parse.quote(object_path, safe="")
    url = f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{encoded_object}?alt=media"

    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def upload_gcs_object(bucket: str, object_path: str, content: bytes, content_type: str = "application/json") -> None:
    """
    Upload bytes to a GCS object using the instance OAuth2 token.

    Args:
        bucket: GCS bucket name, e.g. 'p3dx-tanuh-results'
        object_path: Destination object name within bucket, e.g. 'job-abc/results.json'
        content: Raw bytes to upload
        content_type: MIME type for the object
    """
    token = get_oauth_token()
    encoded_object = urllib.parse.quote(object_path, safe="")
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={encoded_object}"

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
        },
        data=content,
        timeout=60,
    )
    resp.raise_for_status()
