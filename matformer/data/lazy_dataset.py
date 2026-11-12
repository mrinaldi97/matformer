import json
import csv
import math
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple
from torch.utils.data import DataLoader, IterableDataset  
from matformer.matformer_tokenizers import MatformerTokenizer
from matformer.dataclasses.data_dataclasses import DataConfig


DEFAULT_MISSING_LABEL_VALUES = frozenset({"", "null", "none", "nan", "n/a", "na", "nil"})


class LazyIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for lazy, streaming loading of text datasets.
    Supports HuggingFace Hub datasets, JSONL files, and CSV files.

    Args:
        data_config:  DataConfig carrying all dataset settings.
        tokenizer:    Optional, used when ``data_config.tokenize`` is True. Must be a MatformerTokenizer.
        rank:         Rank of the current process in the process group.
        world_size:   Total number of processes in the process group.
        skip_global:  Number of raw rows to skip at iterator start (resume support).
    """

    def __init__(
        self,
        config: DataConfig,
        tokenizer: Optional[MatformerTokenizer] = None,
        rank: int = 0,
        world_size: int = 1,
        skip_global: int = 0,
    ) -> None:
        super().__init__()
        self.data_config  = config
        self.tokenizer    = tokenizer
        self.rank         = rank
        self.world_size   = world_size
        self._skip_global = max(skip_global, self._partition_start())
        self._global_idx: int = self._skip_global   # last processed global raw index; updated while iterating
        if tokenizer is None and getattr(self.data_config, "tokenizer", None) is not None:
            self.tokenizer=self.data_config.tokenizer.instantiate()

        # Label handling state (see _resolve_labels / _full_label_scan)
        self._length: Optional[int] = None
        self._label2id: Optional[Dict[str, int]] = None    # canonical (string-normalized) view of config.label2id
        self._id2label: Dict[int, str] = {}
        self._labels_resolved: bool = False
        self._full_scan_done: bool = False
        self._scan_world_size: Optional[int] = None
        self._scan_valid_by_index: Optional[List[int]] = None
        self._scan_total_rows: int = 0
        self._scan_missing_rows: int = 0
        self._label_counts: Optional[Counter] = None
        self._skipped_missing: int = 0

    @property
    def _classification_labels(self) -> bool:
        """True when a label column is required by the task: missing labels then drop rows."""
        return bool(self.data_config.label_column) and self.data_config.task in ("sentence-level", "regression")

    def _partition_start(self) -> int:
        start = getattr(self.data_config, "partition_start", None)
        return 0 if start is None else start

    def _partition_end(self) -> Optional[int]:
        end = getattr(self.data_config, "partition_end", None)
        head = self.data_config.head_samples
        if head is not None:
            head_end = self._partition_start() + head
            end = head_end if end is None else min(end, head_end)
        return end

    def _missing_values_set(self) -> frozenset:
        extra = getattr(self.data_config, "missing_label_values", None) or []
        return DEFAULT_MISSING_LABEL_VALUES.union(str(v).strip().lower() for v in extra)

    def _is_missing_label(self, value: Any) -> bool:
        """A label is missing when it is None, an empty string, or one of the configured
        placeholders ('null', 'none', 'NaN', ...), which is how CSVs usually spell holes."""
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, str):
            return value.strip().lower() in self._missing_values_set()
        if isinstance(value, (list, tuple)):
            return len(value) == 0
        return False

    @staticmethod
    def _label_key(value: Any) -> str:
        """Canonical string form of a label, so that YAML keys (0, 1) and CSV values ('0', '1')
        meet on the same ground inside label2id."""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _normalize_label2id(self, mapping: Dict[Any, int]) -> Dict[str, int]:
        normalized: Dict[str, int] = {}
        for key, value in mapping.items():
            normalized[self._label_key(key)] = int(value)
        if len(normalized) != len(mapping):
            print(f"WARNING: 'label2id' for column '{self.data_config.label_column}' has keys that "
                  f"collapse onto each other once stringified: {mapping}. Using {normalized}.")
        return normalized

    def _scan_requested(self) -> bool:
        scan = getattr(self.data_config, "scan_labels", None)
        if scan is True:
            return True
        if scan is False:
            return False
        # Automatic: precompute_len already streams the whole file to count rows, so collecting
        # the labels costs nothing extra.
        return bool(self.data_config.precompute_len)

    def _resolve_labels(self) -> None:
        """
        Make ``self._label2id`` usable, inferring it from the data when the user did not provide it.
        A missing mapping can only be recovered by reading every label, so in that case a full pass
        is mandatory and is triggered even when the user asked not to do it (with a loud warning).
        """
        cfg = self.data_config
        if self._labels_resolved or not cfg.label_column or cfg.task == "token-level":
            return
        if cfg.task == "lm":
            # Labels are optional baggage for lm: honour a provided mapping, never force a pass.
            if cfg.label2id:
                self._label2id = self._normalize_label2id(cfg.label2id)
                self._id2label = {v: k for k, v in self._label2id.items()}
            self._labels_resolved = True
            return
        if cfg.label2id:
            self._label2id = self._normalize_label2id(cfg.label2id)
            self._id2label = {v: k for k, v in self._label2id.items()}
            self._labels_resolved = True
            if self._scan_requested():
                self._full_label_scan()
            return
        if cfg.task == "regression":
            # Float targets need no mapping; a scan is only useful for length / missing counts.
            self._labels_resolved = True
            if self._scan_requested():
                self._full_label_scan()
            return
        self._full_label_scan()
        self._labels_resolved = True

    def _full_label_scan(self) -> None:
        """
        One sequential pass over the rows of this partition: counts the valid rows (so the length
        can ignore label-less rows), records the label distribution, infers the label mapping when
        absent, and validates every row against ``label2id`` when it was provided.
        """
        cfg = self.data_config
        if self._full_scan_done:
            if self._scan_world_size == self.world_size:
                return
            # Residue buckets belong to the world size they were counted with: re-count otherwise.
        forced = getattr(cfg, "scan_labels", None) is False or (
            getattr(cfg, "scan_labels", None) is None and not cfg.precompute_len)
        if forced:
            print("*" * 88)
            print(f"WARNING: a FULL SCAN of '{cfg.data_root}' is required to resolve the labels in "
                  f"column '{cfg.label_column}'. This can be VERY SLOW on huge datasets.")
            if not cfg.label2id:
                print("         The label set was not provided: it is being inferred from the data. "
                      "Set 'label2id' in the data section to skip this pass.")
            elif not cfg.precompute_len:
                print("         Set 'scan_labels: false' to keep the exact-length/label statistics off.")
            print("*" * 88)
        self._label2id = self._normalize_label2id(cfg.label2id) if cfg.label2id else None
        end_idx = self._partition_end()
        counts: Counter = Counter()
        unseen: set = set()
        missing = 0
        total = 0
        valid_by_index = [0] * self.world_size
        for global_idx, raw_item in enumerate(self._raw_iterator()):
            if end_idx is not None and global_idx >= end_idx:
                break
            total += 1
            raw_label = raw_item.get(cfg.label_column) if isinstance(raw_item, dict) else None
            if self._is_missing_label(raw_label):
                missing += 1
                continue
            key = self._label_key(raw_label)
            counts[key] += 1
            if cfg.task != "regression" and self._label2id is not None and key not in self._label2id:
                unseen.add(key)
            valid_by_index[global_idx % self.world_size] += 1

        if unseen:
            raise ValueError(
                f"Labels in column '{cfg.label_column}' of '{cfg.data_root}' are not defined in "
                f"label2id: {sorted(unseen)}.\nlabel2id keys: {sorted(self._label2id)}"
            )

        if cfg.task != "regression" and self._label2id is None:
            keys = sorted(counts)
            try:
                numeric = {k: int(k) for k in keys}
            except ValueError:
                self._label2id = {k: i for i, k in enumerate(keys)}
            else:
                # Already-numeric labels keep their own ids (0/1 stays 0/1), so a checkpoint
                # trained on a mapped jsonl and this csv agree on the class indices.
                self._label2id = numeric
            self._id2label = {v: k for k, v in self._label2id.items()}
            cfg.label2id = dict(self._label2id)
            cfg.id2label = dict(self._id2label)
            print(f"Inferred label mapping for column '{cfg.label_column}': {self._label2id}")
            if cfg.head_samples is not None or getattr(cfg, "partition_end", None) is not None:
                print(f"WARNING: '{self.data_config.label_column}' classes were inferred on a PARTIAL "
                      f"dataset (head_samples={cfg.head_samples}, partition_end={cfg.partition_end}): "
                      f"classes living outside this window are missing from label2id/num_labels.")

        self._label_counts = counts
        self._scan_total_rows = total
        self._scan_missing_rows = missing
        self._scan_valid_by_index = valid_by_index
        self._scan_world_size = self.world_size
        self._full_scan_done = True
        self._labels_resolved = True
        if missing:
            print(f"Skipped {missing} labels for missing values in column '{cfg.label_column}' "
                  f"({total} rows scanned, {total - missing} usable).")

    def _map_label(self, label: Any) -> Optional[Any]:
        """Turn a raw label into the int (or float, for regression) the model expects."""
        cfg = self.data_config
        if self._is_missing_label(label):
            if self._classification_labels:
                self._skipped_missing += 1
            return None
        self._resolve_labels()
        if cfg.task == "regression":
            try:
                return float(label)
            except (TypeError, ValueError):
                self._skipped_missing += 1
                return None
        key = self._label_key(label)
        if self._label2id is None:
            # Language-modelling runs with an incidental label column: best effort, never fatal.
            try:
                return int(key)
            except ValueError:
                return label
        if key in self._label2id:
            return self._label2id[key]
        raise KeyError(
            f"Label '{key}' found in column '{cfg.label_column}' is not part of label2id "
            f"(known: {sorted(self._label2id)}). Fix 'label2id', or let the dataset infer it by "
            f"removing it / setting 'scan_labels: true'."
        )

    def get_label_maps(self) -> Tuple[Optional[Dict[str, int]], Optional[Dict[int, str]]]:
        """Return the (label2id, id2label) pair in use, once labels are resolved."""
        self._resolve_labels()
        if self._label2id is None:
            return None, None
        return dict(self._label2id), dict(self._id2label)

    def get_label_info(self) -> Tuple[List[str], int]:
        """Return (labels ordered by id, number of labels)."""
        self._resolve_labels()
        if self._label2id is None:
            raise ValueError(f"No label mapping available for column '{self.data_config.label_column}'.")
        id2label = {v: k for k, v in self._label2id.items()}
        return [id2label[i] for i in range(len(id2label))], len(id2label)

    def get_num_labels(self) -> int:
        return self.get_label_info()[1]

    def get_label_distribution(self) -> Dict[int, int]:
        """Return {label id: row count}. Requires the full scan (it is the only thing that sees
        every row); it is performed on demand, with the usual warning."""
        if not self._full_scan_done:
            self._full_label_scan()
        if self._label_counts is None or self._label2id is None:
            raise ValueError(
                "Cannot get label distribution: no label column configured, or labels not resolved."
            )
        distribution: Dict[int, int] = {i: 0 for i in self._label2id.values()}
        for key, count in self._label_counts.items():
            if key in self._label2id:
                distribution[self._label2id[key]] += count
        return distribution

    def get_class_weights(self, strategy: str = "inverse_frequency", normalize: bool = True) -> List[float]:
        distribution = self.get_label_distribution()
        if strategy == "inverse_frequency":
            weights = [1.0 / max(distribution[i], 1) for i in sorted(distribution)]
        elif strategy == "sqrt_inverse_frequency":
            weights = [1.0 / (max(distribution[i], 1) ** 0.5) for i in sorted(distribution)]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        if normalize:
            total = sum(weights)
            weights = [w * len(weights) / total for w in weights]
        return weights

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._resolve_labels()
        raw_iter = self._raw_iterator()

        # Fast-forward to the resume point without any processing
        for _ in range(self._skip_global):
            try:
                next(raw_iter)
            except StopIteration:
                return

        global_idx: int = self._skip_global
        pending: Optional[Dict[str, Any]] = None  # one-item lookahead buffer
        end_idx = self._partition_end()           

        for raw_item in raw_iter:
            if end_idx is not None and global_idx >= end_idx:
                break

            if global_idx % self.world_size == self.rank:
                item = self._process_item(raw_item)
                if item is not None:
                    if pending is not None:
                        yield pending          
                    pending = item
                    self._global_idx = global_idx + 1

            global_idx += 1

        # Flush the last item with the finished signal for distributed training
        if pending is not None:
            pending["worker_has_finished"] = True
            yield pending

        if self._skipped_missing and not self._full_scan_done:
            print(f"Skipped {self._skipped_missing} labels for missing values in column "
                  f"'{self.data_config.label_column}' while iterating.")

    def compute_length(self) -> int:
            """
            Pre-computes the number of items this rank would see over the full dataset.
            For JSONL/CSV this requires a sequential pass over the file (no data is loaded
            into memory). For HuggingFace datasets the split length is read from metadata.

            Returns:
                Item count assigned to this rank.
            """
            if self.data_config.label_column and self.data_config.task != "token-level":
                # A resolved scan already knows how many label-less rows each residue owns, which
                # is the only length consistent with what the iterator will actually yield.
                self._resolve_labels()
                if self._full_scan_done and self._scan_world_size == self.world_size:
                    return self._scan_valid_by_index[self.rank]
                if self._full_scan_done:
                    self._full_label_scan()
                    return self._scan_valid_by_index[self.rank]
            if self.data_config.file_format == "hf":
                from datasets import load_dataset
                ds    = load_dataset(self.data_config.data_root, split=self.data_config.split)
                total = len(ds)
            else:
                total = 0
                with open(self.data_config.data_root, "r", encoding="utf-8") as f:
                    if self.data_config.file_format == "csv":
                        total = sum(1 for _ in csv.DictReader(f))
                    elif self.data_config.file_format in ("conll", "conllu"):
                        total = sum(1 for _ in self._conll_raw_iterator())
                    else:  # jsonl
                        total = sum(1 for line in f if line.strip())

            start = min(self._partition_start(), total)
            end   = self._partition_end()
            end   = total if end is None else min(end, total)
            end   = max(end, start)

            def _count_below(x: int) -> int:
                return max(0, (x - self.rank + self.world_size - 1) // self.world_size)

            return _count_below(end) - _count_below(start)
    def __len__(self) -> int:
        """Return the number of items this rank will process."""
        if self._length is None:
            self._length = self.compute_length()
        return self._length
    def get_state_dict(self) -> Dict[str, Any]:
            """
            Returns a minimal checkpoint dict.
            The single ``"global_idx"`` value is sufficient to reproduce the exact
            resume point as long as world_size is unchanged.

            Returns:
                Dict with keys ``"global_idx"`` and ``"saved_world_size"``.
            """
            return {
                "global_idx":       self._global_idx,
                "saved_world_size": self.world_size,
            }

    def restore_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restores the iterator position from a checkpoint state dict.

        Args:
            state_dict: Dict as produced by :meth:`get_state_dict`.
        """
        self._skip_global = max(state_dict["global_idx"], self._partition_start())
        self._global_idx  = self._skip_global


    def _raw_iterator(self) -> Iterator[Dict[str, Any]]:
        fmt = self.data_config.file_format
        if fmt == "hf":
            yield from self._hf_raw_iterator()
        elif fmt in ("jsonl", "csv"):
            yield from self._file_raw_iterator()
        elif fmt in ("conll", "conllu"):
            yield from self._conll_raw_iterator()
        else:
            raise ValueError(
                f"Unsupported file_format '{fmt}'. Choose one of: 'hf', 'jsonl', 'csv', 'conll', 'conllu'."
            )

    def _hf_raw_iterator(self) -> Iterator[Dict[str, Any]]:
        """
        Streams rows from a HuggingFace dataset in streaming mode (no full download).
        Uses ``data_config.data_root`` as the dataset path/name and
        ``data_config.split`` as the split name.

        Yields:
            Raw dataset rows as dicts.
        """
        from datasets import load_dataset
        ds = load_dataset(self.data_config.data_root, split=self.data_config.split, streaming=True)
        yield from ds

    def _file_raw_iterator(self) -> Iterator[Dict[str, Any]]:
        """
        Streams rows from a JSONL or CSV file at ``data_config.data_root``.
        Empty lines in JSONL files are silently skipped.
        CSV files must have a header row (parsed via :class:`csv.DictReader`).

        Yields:
            Raw rows as dicts.
        """
        with open(self.data_config.data_root, "r", encoding="utf-8") as f:
            if self.data_config.file_format == "csv":
                yield from csv.DictReader(f)
            else:  # jsonl
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def _conll_raw_iterator(self) -> Iterator[Dict[str, Any]]:
        """
        Streams sentences from a CoNLL/CoNLL-U file, one sentence at a time.
        Yields dicts shaped like {text_column: [tokens], label_column: [tags]}
        """
        is_conllu = self.data_config.file_format == "conllu"
        tokens: List[str] = []
        labels: List[str] = []
        with open(self.data_config.data_root, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    if tokens:
                        yield {self.data_config.text_column: tokens, self.data_config.label_column: labels}
                        tokens, labels = [], []
                    continue
                parts = line.split("\t")
                if "-" in parts[0]:          # skip CoNLL-U multiword tokens
                    continue
                tokens.append(parts[1])
                labels.append(parts[3] if is_conllu else parts[4])
        if tokens:
            yield {self.data_config.text_column: tokens, self.data_config.label_column: labels}                        

    def _tokenize_and_align(self, words: List[str], labels: List[Any]) -> tuple:
        input_ids: List[int] = []
        aligned_labels: List[Any] = []
        for word, lab in zip(words, labels):
            word_ids = self.tokenizer.encode(word, add_bos=False, add_eos=False)["input_ids"][0]
            input_ids.extend(word_ids)
            aligned_labels.append(lab)
            aligned_labels.extend([-100] * (len(word_ids) - 1))
        return input_ids, aligned_labels

    def _process_item(self, raw_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
            Tokenizes (if requested) and packages a raw dataset row into the dict
            format expected by ``MatformerDataModule.collate_fn``.

            If ``data_config.tokenize`` is True the text at ``text_column`` is passed
            through the tokenizer and the result stored as ``input_ids``.
            The ids are then read from ``input_ids_column`` (default: ``"input_ids"``)
            and truncated to ``max_seq_len`` if necessary.

            Args:
                raw_item: A raw row dict from the dataset.

            Returns:
                Processed dict with keys ``"object"``, ``"metadata"``, and
                ``"worker_has_finished"``; or ``None`` when the item lacks token ids.
        """
        item: Dict[str, Any] = dict(raw_item)
        cfg = self.data_config
        label = item.get(cfg.label_column) if cfg.label_column else None

        if cfg.label2id is not None and label is not None:
            if cfg.task == "token-level":
                label = [cfg.label2id[l] if isinstance(l, str) else l for l in label]
        if cfg.task != "token-level" and cfg.label_column:
            label = self._map_label(label)
            if label is None and self._classification_labels: #Skip missing labels
                return None
        if cfg.task == "token-level":
            if cfg.tokenize:
                assert self.tokenizer is not None, "A tokenizer must be provided when tokenize=True."
                input_ids, label = self._tokenize_and_align(item[cfg.text_column], label)
            else:
                input_ids = item.get(cfg.input_ids_column)
        else:
            if cfg.tokenize:
                assert self.tokenizer is not None, "A tokenizer must be provided when tokenize=True."
                tok_out = self.tokenizer.encode(
                    item[cfg.text_column], add_eos=cfg.add_eos, add_bos=cfg.add_bos, max_length=getattr(cfg,"max_seq_len",None)
                )
                item[cfg.input_ids_column] = tok_out["input_ids"][0]
            input_ids = item.get(cfg.input_ids_column)

        if input_ids is None:
            return None

        if getattr(cfg,"max_seq_len",None) is not None:
            input_ids = input_ids[: cfg.max_seq_len]
            if cfg.task == "token-level" and label is not None:
                label = label[: cfg.max_seq_len]

        result: Dict[str, Any] = {"object": input_ids, "worker_has_finished": False}
        if label is not None:
            result["labels"] = label

        if cfg.keep_metadata:
            metadata = {k: v for k, v in item.items() if k != cfg.input_ids_column}
            if cfg.keep_original_text and cfg.text_column:
                metadata["text"] = item.get(cfg.text_column, "")
            result["metadata"] = metadata
        else:
            result["metadata"] = None

        return result


