# matformer/data/annotation.py
"""Annotation sinks: the opt-in alternative to the run database.

When a task is driven over a dataset with ``--input``, its per-item records go to one JSONL file
or to an mdat and its summary to ``<output>.summary.jsonl``. These are plain writers, not a second
persistence system: the task hands them the same records it would otherwise log to SQLite.
"""
import json
from pathlib import Path

from matformer.utils.run_db import jsonable


class AnnotationWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records = self.path.open("wb")
        self._summary = Path(str(self.path) + ".summary.jsonl").open("wb")
        try:
            import orjson
            self._dumps = orjson.dumps
        except ImportError:
            self._dumps = lambda obj: json.dumps(obj, ensure_ascii=False).encode()

    def records(self, records) -> None:
        for record in records:
            self._records.write(self._dumps(jsonable(record)) + b"\n")
        self._records.flush()

    def summary(self, summary) -> None:
        self._summary.write(self._dumps(jsonable(summary)) + b"\n")
        self._summary.flush()

    def close(self) -> None:
        self._records.close()
        self._summary.close()


class MdatAnnotationWriter:
    """Annotation sink that writes an mdat instead of a JSONL file.

    Records are appended in input order. With ``keys_only=True`` only the
    predictions (plus the source ``key``) are stored, in a metadata-only mdat:
    annotating a huge database then does not copy its raw documents. Otherwise
    the source metadata (including the original text) is preserved and the
    predictions are added to it.
    """

    def __init__(self, path, data_column="text", keys_only=False, key_field="_mdat_key",
                 map_sizes=None, overwrite=True):
        import shutil
        from matformer.data.mdat.mdat import mdat_writer
        self.path = Path(path)
        self.data_column = None if keys_only else data_column
        self.keys_only = keys_only
        self.key_field = key_field
        if overwrite and self.path.exists() and self.path.is_dir() and (
                (self.path / "mdat_container.db").exists() or (self.path / "mdat.db").exists()):
            shutil.rmtree(self.path)
        self._writer = mdat_writer(str(self.path), data_column=self.data_column,
                                   keys_only=keys_only, map_sizes=map_sizes or {})
        self._summary = Path(str(self.path) + ".summary.jsonl").open("wb")
        try:
            import orjson
            self._dumps = orjson.dumps
        except ImportError:
            self._dumps = lambda obj: json.dumps(obj, ensure_ascii=False).encode()

    def _payload(self, record):
        record = jsonable(record)
        attributes = record.pop("attributes", None) or {}
        if not isinstance(attributes, dict):
            attributes = {}
        metadata = dict(attributes.get("metadata") or {})
        source_key = metadata.pop(self.key_field, None)
        if source_key is None:
            source_key = record.pop("id", None)
            if source_key is not None:
                source_key = int(source_key) - 1
        else:
            record.pop("id", None)
        annotations = {key: value for key, value in record.items()
                       if key not in ("input_ids", "worker_has_finished")}
        if self.keys_only:
            payload = {"key": source_key}
            payload.update(annotations)
            return payload
        payload = metadata
        if self.data_column and self.data_column not in payload and "text" in payload:
            payload[self.data_column] = payload.pop("text")
        payload.setdefault("key", source_key)
        payload.update(annotations)
        return payload

    def records(self, records) -> None:
        for record in records:
            self._writer.write(self._payload(record))

    def summary(self, summary) -> None:
        self._summary.write(self._dumps(jsonable(summary)) + b"\n")
        self._summary.flush()

    def close(self) -> None:
        self._writer.close()
        self._summary.close()

