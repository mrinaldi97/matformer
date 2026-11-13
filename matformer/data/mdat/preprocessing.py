"""Input iterators used to feed raw data into a dataset (convert_to_mdat).

Each iterator yields python dictionaries, one per document; the consuming code
picks the document body out of them using its ``data_key``.
"""
from __future__ import annotations

import json
import os

import orjson
from tqdm import tqdm

from .lmdb import LMDBDataset


class _NullProgressBar:
    """No-op stand-in for tqdm, so iterators can share one code path whether
    the progress bar is enabled or not. Usable both as an iterable wrapper
    (``for i in ProgressBar(range(n))``) and as a context manager
    (``with ProgressBar(total=n) as pbar``)."""

    def __init__(self, iterable=None, *args, **kwargs):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable) if self._iterable is not None else iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


def HuggingFaceIterator(dataset_path, logger, dataset_args, dataset_config='', dataset_split='train', progress_bar=True):
    from datasets import load_dataset
    ds = load_dataset(dataset_path, dataset_config)[dataset_split]
    for item in tqdm(ds, disable=not progress_bar):
        yield item


def JSONIterator(json_path, logger, dataset_args={}, progress_bar=True, batch_size_bytes=16 * 1024 * 1024):
    file_size = os.path.getsize(json_path)

    ProgressBar = tqdm if progress_bar else _NullProgressBar

    with open(json_path, 'rb', buffering=batch_size_bytes) as f:
        with ProgressBar(total=file_size, unit='B', unit_scale=True, desc="Processing JSONL file...") as pbar:
            i = 0
            while True:
                batch_of_lines = f.readlines(batch_size_bytes)
                if not batch_of_lines:
                    break

                for row_bytes in batch_of_lines:
                    try:
                        data = orjson.loads(row_bytes)
                    except Exception as e:
                        try:
                            logger.warning(f"Row: {i} orjson failed: {e}, falling back to json.")
                            data = json.loads(row_bytes)
                        except Exception as e:
                            logger.error(f"Row {i} Failed to parse JSON row: {e}")
                            data = None

                    if data is not None:
                        pbar.update(len(row_bytes))
                        yield data

                    i += 1


def AtlasIterator(path, logger, dataset_args={}, progress_bar=True):
    from torch_atlas_ds import AtlasDataset
    db = AtlasDataset(path)
    ProgressBar = tqdm if progress_bar else _NullProgressBar
    for i in ProgressBar(range(len(db))):
        try:
            yield db[i]
        except Exception as e:
            logger.error(f"Error reading Atlas item {i}: {e}")
            yield None


def LMDBIterator(lmdb_path, logger, dataset_args={}, progress_bar=True):
    db = LMDBDataset.open(lmdb_path)
    ProgressBar = tqdm if progress_bar else _NullProgressBar
    for i in ProgressBar(range(len(db))):
        try:
            yield db[i]
        except Exception as e:
            logger.error(f"Error reading LMDB item {i}: {e}")
            yield None
