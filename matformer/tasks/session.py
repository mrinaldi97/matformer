"""The driving layer of Matformer: TaskSession binds a loaded model to its tasks and feeds it,
Stats accounts for what a run cost, TaskConsole is the cmd2 shell run_task.py opens."""
from __future__ import annotations

import dataclasses
import inspect
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from cmd2 import Cmd, Cmd2ArgumentParser, with_argparser

from matformer.matformer_registry import registry
from matformer import sweeps
from matformer.dataclasses.tensors_dataclasses import PaddedTensor, TensorDC
from matformer.tasks.task import PRESENTATIONS, split_result
from matformer.utils.schema import field_docs


def describe(config) -> str:
    docs = field_docs(type(config))
    width = max(len(f.name) for f in dataclasses.fields(config))
    lines = [f"{type(config).__name__} '{config.name}'", '']
    lines += [f"  {f.name:<{width}}  {str(getattr(config, f.name))[:70]:<24}  {docs.get(f.name, '')}"
              for f in dataclasses.fields(config)]
    return '\n'.join(lines)


def render(value) -> str:
    if torch.is_tensor(value):
        value = value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return '  '.join(f"{key}={render(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        if len(value) > 6:
            if all(isinstance(item, (int, float)) for item in value):
                picks = sorted(range(len(value)), key=value.__getitem__, reverse=True)[:3]
                return '[' + ', '.join(f"#{i}:{render(value[i])}" for i in picks) + f', …/{len(value)}]'
            value = list(value[:3]) + ['…']
        return '[' + ', '.join(render(item) for item in value) + ']'
    return str(value)


def format_summary(summary, hide=('model', 'tokenizer', 'data', 'task', 'training')) -> str:
    summary = {key: value for key, value in (summary or {}).items() if key not in hide}
    return '\n'.join(f"  {key}: {render(value)}" for key, value in summary.items()) or '  (no summary)'


def format_rows(rows, stream=False) -> str:
    return '\n'.join(render(row) for row in rows if not (stream and isinstance(row, str)))


def token_count(batch) -> int:
    """How many real tokens a batch holds (padding excluded)."""
    mask = getattr(batch, 'padding_mask', None)
    if mask is not None:
        return int((~mask).sum())
    return int((batch.tensor if isinstance(batch, TensorDC) else batch).numel())


def gib(bytes_: Optional[float]) -> str:
    return f"{bytes_ / 2**30:.2f} GiB" if bytes_ is not None else "-"


@dataclass
class Stats:
    """What the session did so far. The seconds are the wall clock around the task call, the
    token counts and the per-phase speeds are reported by the task itself. `report` describes the
    last run, `full` describes all of them together with the memory of the process."""
    runs: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    seconds: float = 0.0
    last: dict = field(default_factory=dict)

    def add(self, in_tokens=0, out_tokens=0, seconds=0.0, metrics=None) -> None:
        self.runs += 1
        self.in_tokens += in_tokens
        self.out_tokens += out_tokens
        self.seconds += seconds
        self.last = {'in': in_tokens, 'out': out_tokens, 'seconds': seconds, 'metrics': metrics or {}}

    @staticmethod
    def rate(tokens, seconds) -> str:
        return f"{tokens / seconds:.1f} tok/s" if seconds else "-"

    @property
    def tokens_per_sec(self) -> float:
        return self.out_tokens / self.seconds if self.seconds else 0.0

    def report(self) -> str:
        if not self.last:
            return "no run yet"
        bits = [f"{self.last['in']} in", f"{self.last['out']} out tok",
                f"{self.last['seconds'] * 1e3:.0f} ms"]
        if self.last['out']:
            bits.append(f"{self.rate(self.last['out'], self.last['seconds'])}")
        for key, value in self.last['metrics'].items():
            if key.endswith('_tokens') and f"{key[:-7]}_seconds" in self.last['metrics']:
                name = key[:-7]
                bits.append(f"{name} {self.rate(value, self.last['metrics'][f'{name}_seconds'])}")
        return ' · '.join(bits)

    def full(self, model) -> str:
        info, cache = memory(model), getattr(model, 'cache', None)
        lines = [f"  runs        {self.runs}",
                 f"  tokens      {self.in_tokens} in / {self.out_tokens} out",
                 f"  time        {self.seconds:.2f} s  ({self.tokens_per_sec:.1f} tok/s)"]
        for key, value in (self.last.get('metrics') or {}).items():
            lines.append(f"  {key:<11}  {value:.1f}" if isinstance(value, float)
                         else f"  {key:<11}  {value}")
        lines += ['',
                  f"  parameters  {info['parameters'] / 1e6:.1f} M",
                  f"  weights     {gib(info['weights'])}"]
        if 'allocated' in info:
            lines += [f"  cuda        {gib(info['allocated'])} allocated",
                      f"  reserved    {gib(info['reserved'])}  (peak {gib(info['peak'])})"]
        if cache is not None:
            kv = cache.kv_cache_report()
            lines.append(f"  kv cache    {kv if kv else 'none (use_cache is off)'}")
            lines.append(f"  misc cache  {cache.get_cache_size()}")
        return '\n'.join(lines)


def memory(model) -> dict:
    """Size of the loaded model and of the CUDA allocator around it."""
    params = sum(p.numel() for p in model.parameters())
    out = {"parameters": params, "weights": sum(p.numel() * p.element_size() for p in model.parameters())}
    if torch.cuda.is_available():
        out.update(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                   peak=torch.cuda.max_memory_allocated())
    return out


class StreamDecoder:

    def __init__(self, tokenizer, prompt_ids, eos_token_id=None):
        self.tokenizer, self.eos_token_id = tokenizer, eos_token_id
        self.ids = list(prompt_ids)
        self.printed = len(tokenizer.decode(self.ids))

    def push(self, token_id) -> str:
        if token_id == self.eos_token_id:
            return ''
        self.ids.append(token_id)
        text = self.tokenizer.decode(self.ids)
        new_chars, self.printed = text[self.printed:], len(text)
        return new_chars


class TaskSession:

    INPUT_FORMATS = {'jsonl': 'jsonl', 'json': 'jsonl', 'csv': 'csv', 'conll': 'conll',
                     'conllu': 'conllu', 'mdat': 'mdat'}

    def __init__(self, model, configs, task=None, overrides=None, num_devices=1, source=None,
                 presentation=None):
        self.model, self.configs, self.num_devices = model, configs, num_devices
        self.source = source
        self.stats = Stats()
        self._task = None
        self.use(task if task is not None else self.suggested_task())
        if overrides:
            self.configs.set_override(overrides)
            self._task = None
        if presentation:
            self.configure('presentation', presentation)

    def suggested_task(self) -> str:
        group = getattr(self.configs, 'task', None)
        if group is not None and group.objects:
            return group.default
        variant = getattr(getattr(self.configs, 'model', None), '_matformer_variant', '')
        return 'evaluation' if variant in ('classification', 'multiprobe') else 'generation'

    def new_entry(self, task_type) -> dict:
        entry = {'type': task_type, 'name': task_type}
        if task_type == 'evaluation':
            scope = getattr(getattr(self.configs, 'model', None), 'scope', None)
            entry['learning_signal'] = 'token_classification' if scope == 'token' else 'sequence_classification'
        return entry

    @staticmethod
    def task_types() -> list:
        return list(registry.get_registry_info(category='task')['task']['variants'])

    @property
    def task_config(self):
        return getattr(self.configs.task, self.task_name)

    def use(self, name) -> str:
        group = getattr(self.configs, 'task', None)
        if group is None or name not in group.objects:
            if name not in self.task_types():
                raise ValueError(f"'{name}' is neither a configured task "
                                 f"{group.list_configs() if group else []} nor a task type {self.task_types()}")
            self.configs.import_config({'task': [self.new_entry(name)]})
            self.configs.resolve_references()
        self.configs.task.set_default(name)
        self.task_name, self._task = name, None
        return name

    def configure(self, path, value=None) -> None:
        if '.' not in path or path.split('.')[0] not in self.configs.objects:
            valid = {f.name for f in dataclasses.fields(self.task_config)}
            if path not in valid:
                raise ValueError(f"'{path}' is not a parameter of {type(self.task_config).__name__}. "
                                 f"Valid: {sorted(valid)}")
            path = f"task.{self.task_name}.{path}"
        self.configs.set_override(f"{path}={value}")
        self._task = None

    @property
    def task(self):
        if self._task is None:
            self._task = self.task_config.instantiate(model=self.model, num_devices=self.num_devices)
        return self._task

    @property
    def tokenizer(self):
        tokenizer = getattr(self.model, 'tokenizer', None)
        if tokenizer is None:
            raise ValueError("This model has no tokenizer: load one with --tokenizer.")
        return tokenizer

    def data_rules(self):
        rules = getattr(self.task_config, 'data', None)
        if rules is not None:
            return rules
        group = getattr(self.configs, 'data', None)
        if group is not None and group.objects:
            return getattr(group, group.default)
        return registry.get_definition('data', 'lazy').from_dict({})

    def batch(self, texts):
        rules = self.data_rules()
        words = bool(getattr(self.task, 'word_level', False))
        texts = [str(text).split() for text in texts] if words else [str(text) for text in texts]
        encoded = self.tokenizer.encode(
            text=texts, is_split_into_words=words, return_word_ids=words,
            add_bos=rules.add_bos, add_eos=rules.add_eos,
            max_length=rules.max_seq_len or getattr(self.configs.model, 'max_position_embeddings', None))
        rows = [torch.tensor(ids, dtype=torch.long, device=self.model.device) for ids in encoded['input_ids']]
        batch = PaddedTensor.from_rows(rows, pad_value=getattr(self.tokenizer.config, 'pad_token_id', None) or 0)
        if words:
            batch.attributes.update(words=texts, word_ids=encoded['word_ids'])
        return batch.unpad() if rules.varlen_strategy == 'unpadding' else batch

    @property
    def wants_stream(self) -> bool:
        return bool(self.task_config.get('stream', False))

    def feed(self, text, stream=False, save=False, streamer=None, **kwargs) -> list:
        """Run one text through the task. Pass a custom `streamer(next_tokens, step)` to receive
        the tokens somewhere other than stdout (GUI, server): the same code path the TUI uses."""
        batch = self.batch([text])
        streamer = streamer or (self.streamer(batch) if stream else None)
        started = time.perf_counter()
        rows = split_result(self.task(batch, save=save, stream_callback=streamer, **kwargs), len(batch))
        metrics = dict(self.task.metrics)
        self.stats.add(metrics.pop('in_tokens', token_count(batch)), metrics.pop('new_tokens', 0),
                       time.perf_counter() - started, metrics)
        return rows

    def streamer(self, batch):
        ids = batch.pad().pad_right().tensor[0].tolist()
        decoder = StreamDecoder(self.tokenizer, ids, getattr(self.tokenizer.config, 'eos_token_id', None))

        def emit(next_tokens, step):
            sys.stdout.write(decoder.push(next_tokens[0].item()))
            sys.stdout.flush()
        return emit

    def status(self) -> str:
        """The last run, plus whatever the task wants to show: the console prints this after every
        request, so the cost of the conversation is never more than a glance away."""
        bits = [self.stats.report(), self.task.status, self.kv_status()]
        return '  '.join(bit for bit in bits if bit)

    def kv_status(self) -> str:
        """KV cache footprint, only once it exists: an off use_cache should not clutter the line."""
        cache = getattr(self.model, 'cache', None)
        if cache is None or not cache.storage.get('paged_kv'):
            return ""
        kv = cache.storage['paged_kv']
        return f"kv {kv.bytes / 2**20:.0f} MiB ({kv.used_tokens}/{kv.capacity_tokens} tok)"

    def run_text(self, text, save=False) -> str:
        """One text through the current task, rendered the way the task likes to be read."""
        stream = self.wants_stream
        rows = self.feed(text, stream=stream, save=save)
        rendered = self.task.present(rows)
        if rendered is None:
            rendered = format_rows(rows, stream)
        return ('\n' if stream else '') + '\n'.join(bit for bit in (rendered, self.status()) if bit)

    def model_info(self) -> str:
        config, lines = getattr(self.configs, 'model', None), []
        lines.append(f"  source      {self.source or f'untrained {type(self.model.model).__name__}'}")
        lines.append(f"  model       {type(self.model.model).__name__}"
                     + (f" ({config._matformer_variant})" if config is not None else ''))
        if config is not None:
            for key in ('name', 'scope', 'num_classes', 'hidden_size', 'num_hidden_layers', 'vocab_size',
                        'max_position_embeddings', 'is_causal', 'training_objective'):
                value = getattr(config, key, None)
                if value not in (None, []):
                    lines.append(f"  {key:<11}  {value}")
            label2id = getattr(config, 'label2id', None)
            if label2id:
                lines.append(f"  labels      " + ', '.join(
                    f"{i}={name}" for name, i in sorted(label2id.items(), key=lambda kv: kv[1])))
        lines.append(f"  runtime     {self.model.runtime_mode} on {self.model.device}, "
                     f"plugins: {list(getattr(self.model, 'plugins', {}) or {})}")
        tokenizer = getattr(self.model, 'tokenizer', None)
        if tokenizer is not None:
            lines.append(f"  tokenizer   {getattr(tokenizer.config, 'pretrained_name', None)}")
            lines.append('  specials    ' + '  '.join(
                f"{key}={getattr(tokenizer.config, key, None)}"
                for key in ('bos_token_id', 'eos_token_id', 'pad_token_id', 'mask_token_id')))
        lines += ['', self.stats.full(self.model)]
        return '\n'.join(lines)

    def run(self) -> dict:
        return self._run()[1]

    def _run(self):
        task = self.task
        try:
            summary = task.run()
        finally:
            self._task = None
        return task, summary

    def annotate(self, input_path, output_path=None, column='text', label=None,
                 limit=None, batch_size=None, output_format=None, keys_only=False,
                 mdat_dataset=None) -> dict:
        suffix = os.path.splitext(input_path)[1].lstrip('.').lower()
        is_mdat = os.path.isdir(input_path) or suffix == 'mdat'
        if '::' in input_path:
            input_path, mdat_dataset = input_path.rsplit('::', 1)
            suffix = os.path.splitext(input_path)[1].lstrip('.').lower()
            is_mdat = os.path.isdir(input_path) or suffix == 'mdat' or mdat_dataset is not None
        if not is_mdat and suffix not in self.INPUT_FORMATS:
            raise ValueError(f"Cannot read '{input_path}': choose one of {sorted(self.INPUT_FORMATS)} files.")
        rules = self.data_rules()
        if output_format is None:
            output_format = 'mdat' if is_mdat or (output_path or '').rstrip('/').endswith('.mdat') else 'jsonl'
        if output_format not in ('jsonl', 'mdat'):
            raise ValueError(f"Unknown annotation output format '{output_format}'")
        if keys_only and output_format != 'mdat':
            raise ValueError("keys_only=True is only available with an mdat output")
        if output_path is None:
            stem = input_path.rstrip('/')
            output_path = (stem if is_mdat else os.path.splitext(stem)[0]) + \
                ('.annotated.mdat' if output_format == 'mdat' else '.annotated.jsonl')
        if is_mdat and output_format == 'mdat' and \
                os.path.abspath(output_path) == os.path.abspath(input_path):
            raise ValueError("The annotation output cannot be the input mdat itself; choose another path.")
        if self.configs.active_sweep is not None:   # every point of a sweep annotates its own file
            output_path = sweeps.scope(output_path, self.configs.active_sweep.name)
        file_format = 'mdat' if is_mdat else self.INPUT_FORMATS[suffix]
        data_config = {'name': 'annotate', 'type': 'lazy', 'file_format': file_format,
                       'data_root': input_path, 'text_column': column, 'label_column': label,
                       'task': 'sentence-level' if label else 'lm', 'tokenize': True,
                       'keep_metadata': True, 'keep_original_text': True, 'precompute_len': True,
                       'max_seq_len': rules.max_seq_len, 'varlen_strategy': rules.varlen_strategy,
                       'batch_size': batch_size or rules.batch_size, 'head_samples': limit}
        if file_format == 'mdat':
            data_config['mdat_dataset'] = mdat_dataset
        self.configs.import_config({'data': [data_config]})
        for path, value in {'ref_data': 'annotate', 'save_batch': True,
                            'save_attributes': True, 'save_probs': True, 'flatten_dict': True}.items():
            self.configure(f"task.{self.task_name}.{path}", value)
        if output_format == 'mdat':
            from matformer.data.annotation import MdatAnnotationWriter
            writer = MdatAnnotationWriter(output_path, data_column=column, keys_only=keys_only)
        else:
            from matformer.data.annotation import AnnotationWriter
            writer = AnnotationWriter(output_path)
        task = self.task
        task.record_sink, task.summary_sink = writer.records, writer.summary
        try:
            summary = task.run()
        finally:
            writer.close()
            self._task = None
        return {'output': output_path, 'rows': task.single_item_id, **summary}


ANNOTATE_PARSER = Cmd2ArgumentParser(description="Annotate a file or mdat with the current task.")
ANNOTATE_PARSER.add_argument('input', completer=Cmd.path_complete,
                             help="jsonl/csv/conll file or mdat folder holding one text per row; "
                                  "use PATH::DATASET for a multi-dataset container")
ANNOTATE_PARSER.add_argument('-o', '--output', help="output file/folder (default: <input>.annotated.jsonl/.mdat)")
ANNOTATE_PARSER.add_argument('-c', '--column', default='text', help="column holding the text (default: text)")
ANNOTATE_PARSER.add_argument('-l', '--label', help="column holding the gold label, to score the model")
ANNOTATE_PARSER.add_argument('-n', '--limit', type=int, help="annotate only the first N rows")
ANNOTATE_PARSER.add_argument('-b', '--batch_size', type=int, help="rows per forward pass")
ANNOTATE_PARSER.add_argument('-f', '--format', dest='output_format', choices=['jsonl', 'mdat'],
                             help="output format (default: mdat for mdat input, jsonl otherwise)")
ANNOTATE_PARSER.add_argument('--keys-only', action='store_true',
                             help="mdat output holding only key + annotations, not a copy of the documents")
ANNOTATE_PARSER.add_argument('--dataset', help="dataset name when the input is a multi-dataset mdat container")

PRESENT_PARSER = Cmd2ArgumentParser(description="How predictions are shown.")
PRESENT_PARSER.add_argument('mode', choices=list(PRESENTATIONS), nargs='?',
                            help="probs: label + confidence (default). argmax: label only. "
                                 "full: everything the task returned.")


class TaskConsole(Cmd):
    """A plain '>>>' prompt: type a text to run the current task on it. Everything the shell
    knows about comes from the session and from the task, which own the state."""

    def __init__(self, session, **kwargs):
        kwargs.setdefault('allow_cli_args', False)
        super().__init__(persistent_history_file=os.path.expanduser('~/.matformer_history'),
                         terminators=[], **kwargs)
        self.session = session
        self.prompt = ">>> "

    def do_tasks(self, _):
        for name, config in self.session.configs.task:
            marker = '*' if name == self.session.task_name else ' '
            self.poutput(f" {marker} {name:<16} {config._matformer_variant:<14} "
                         f"data={getattr(getattr(config, 'data', None), 'name', None)}")
        self.poutput(f"\n  available types: {', '.join(self.session.task_types())}   (use <type> to add one)")

    def do_use(self, name):
        try:
            self.session.use(name.strip())
        except ValueError as problem:
            self.poutput(problem)
            return
        self.poutput(f"task: {self.session.task_name}")

    def do_show(self, arg):
        if not arg.split():
            self.poutput(describe(self.session.task_config))
            return
        section = getattr(self.session.configs, arg.split()[0], None)
        if section is None:
            self.poutput(f"the configs have no '{arg.split()[0]}' section")
        else:
            self.poutput(describe(section) if dataclasses.is_dataclass(section) else section)

    def do_set(self, line):
        path, _, value = line.partition(' ')
        if not path:
            self.do_show('')
            return
        try:
            self.session.configure(path, value.strip())
        except ValueError as problem:
            self.poutput(problem)
            return
        self.poutput(f"{path} = {value.strip()}")

    def complete_set(self, text, line, begidx, endidx):
        return [f.name for f in dataclasses.fields(self.session.task_config) if f.name.startswith(text)]

    @with_argparser(PRESENT_PARSER)
    def do_present(self, args):
        if not args.mode:
            self.poutput(f"presentation: {self.session.task_config.presentation}")
            return
        try:
            self.session.configure('presentation', args.mode)
        except ValueError as problem:
            self.poutput(problem)
            return
        self.poutput(f"presentation: {args.mode}")

    def do_run(self, _):
        self.poutput(format_summary(self.session.run()))

    @with_argparser(ANNOTATE_PARSER)
    def do_annotate(self, args):
        summary = self.session.annotate(args.input, output_path=args.output, column=args.column,
                                        label=args.label, limit=args.limit, batch_size=args.batch_size,
                                        output_format=args.output_format, keys_only=args.keys_only,
                                        mdat_dataset=args.dataset)
        self.poutput(f"{summary.pop('rows')} rows -> {summary.pop('output')}")
        self.poutput(format_summary(summary))

    def do_say(self, text):
        self.run_text(text)

    def do_model(self, _):
        self.poutput(self.session.model_info())

    def do_stats(self, _):
        self.poutput(self.session.stats.full(self.session.model))

    def do_reset(self, _):
        self.session.task.reset()
        self.poutput(f"{self.session.task_name} reset")

    def _for_task(self, command, args, missing):
        handler = getattr(self.session.task, command, None)
        if handler is None:
            self.poutput(missing.format(task=self.session.task_name))
            return
        self.poutput(handler(*args) or '')

    def do_clear(self, _):
        self._for_task('clear', (), "the '{}' task keeps nothing to clear")

    def do_context(self, _):
        self._for_task('show_context', (), "the '{}' task has no context")

    def do_system(self, text):
        self._for_task('set_system_prompt', (text,), "the '{}' task takes no system prompt")

    def do_template(self, text):
        self._for_task('set_chat_template', (text,), "the '{}' task takes no chat template")

    def default(self, statement):
        self.run_text(getattr(statement, 'raw', statement))

    def run_text(self, text):
        text = (text or '').strip()
        if not text:
            return
        stream = self.session.wants_stream
        try:
            rows = self.session.feed(text, stream=stream)
        except Exception as problem:
            self.poutput(f"the '{self.session.task_name}' task failed: {problem}")
            return
        if stream:
            sys.stdout.write('\n')
        rendered = self.session.task.present(rows)
        if rendered is None:
            rendered = format_rows(rows, stream)
        if rendered:
            self.poutput(rendered)
        if status := self.session.status():
            self.poutput(f"({status})")
