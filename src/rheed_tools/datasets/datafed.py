from __future__ import annotations

"""Optional DataFed convenience wrappers from archived acquisition notebooks."""

import json
from pathlib import Path
from typing import Any


def create_collection(collection_name: str, parent_id: str | None = None):
    """Create a DataFed collection using an installed `datafed` client."""

    api = _datafed_api()
    return api.collectionCreate(collection_name, parent_id=parent_id)


def list_collection_items(collection_id: str, *, max_count: int = 100):
    """Return DataFed collection items without printing notebook side effects."""

    api = _datafed_api()
    return list(api.collectionItemsList(collection_id, count=max_count)[0].item)


def upload_file(file_path: str | Path, parent_id: str, metadata: dict[str, Any] | None = None, *, wait: bool = True) -> str:
    """Upload one file to DataFed and return the created record id."""

    api = _datafed_api()
    path = Path(file_path)
    response = api.dataCreate(path.name, metadata=json.dumps(metadata or {}), parent_id=parent_id)
    record_id = response[0].data[0].id
    api.dataPut(record_id, str(path), wait=wait)
    return str(record_id)


def download_file(file_id: str, output_dir: str | Path, *, wait: bool = True):
    """Download a DataFed file by id into `output_dir`."""

    api = _datafed_api()
    return api.dataGet([file_id], str(output_dir), orig_fname=True, wait=wait)


def update_record_metadata(record_id: str, metadata: dict[str, Any]):
    """Replace a DataFed record's metadata."""

    api = _datafed_api()
    return api.dataUpdate(record_id, metadata=json.dumps(metadata), metadata_set=True)


def _datafed_api():
    try:
        from datafed.CommandLib import API
    except ImportError as exc:
        raise ImportError("DataFed helpers require the optional `datafed` package.") from exc
    return API()
