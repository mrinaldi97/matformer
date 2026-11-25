#!/usr/bin/env python3
"""run_task: drive the tasks of a Matformer model from the command line.

    python3 run_task.py --model checkpoint.ckpt -i
    python3 run_task.py --model hf://Qwen/Qwen3-0.6B --text "Ciao"
    python3 run_task.py --model checkpoint.ckpt --task masking --text "Ciao a tutti"
    python3 run_task.py --model checkpoint.ckpt --input data.jsonl --column text
    python3 run_task.py --cfg exp.yaml --sweeps            # what the sweep would run
    python3 run_task.py --cfg exp.yaml                     # every point of the sweep

Everything that happens after the model is built lives in matformer.tasks.session: this file
decides what to load and whether to open the console, run one text, annotate a file or run every
task declared in the configs -- and, since a config may declare a sweep, how many times to do it.
A `checkpoint` section can name the model too, so a YAML alone can drive a task.
"""
import argparse
import os

import torch

from matformer import sweeps
from matformer.matformer_configs import MatformerConfigs
from matformer.matformer_tokenizers import MatformerTokenizer
from matformer.matformer_wrapper import load_model
from matformer.tasks.session import PRESENTATIONS, TaskConsole, TaskSession, format_summary


def parser():
    argp = argparse.ArgumentParser(
        prog="run_task", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argp.add_argument('--model', default=None,
                      help="model checkpoint; omit to build the model described by the configs, untrained")
    argp.add_argument('--tokenizer', default=None,
                      help="HF tokenizer id; default: the one the checkpoint was trained with")
    argp.add_argument('--cfg', action='append', default=None, metavar='YAML',
                      help="config file(s), merged left to right; each may hold any section")
    argp.add_argument('--override', action='append', default=None, metavar='SECTION.FIELD=VALUE',
                      help="override applied after the YAMLs, ex. task.test.topk=false; "
                           "it can address the task selected with --task, even if new")
    argp.add_argument('--device', default=None, help="cuda/cpu/mps (default: cuda if available)")
    argp.add_argument('--num_devices', type=int, default=1, help="devices for the data loaders")
    argp.add_argument('--task', default=None,
                      help="task entry or task type to drive (default: the first one declared)")
    argp.add_argument('--present', choices=PRESENTATIONS, default=None,
                      help="how a prediction is shown: label with its probability (default), "
                           "the label alone, or everything the task computed")
    argp.add_argument('--attention', default=None, metavar='IMPLEMENTATION',
                      help="attention implementation to build the model with, ex. flashinfer "
                           "(paged KV cache) or sdpa; same as MATFORMER_ATTENTION_NORMAL")
    argp.add_argument('--sweep', default=None, metavar='POINTS',
                      help="only these points of the config's sweep: numbers (1-based) or name "
                           "globs, comma separated. Default: every point of it.")
    argp.add_argument('--sweeps', action='store_true',
                      help="print the sweep the configs declare -- every point, and what it "
                           "changes -- and run nothing")
    one_shot = argp.add_argument_group('one-shot modes (default: run every task of the config)')
    one_shot.add_argument('-i', '--interactive', action='store_true', help="interactive console")
    one_shot.add_argument('--text', help="run the task on this text and print the result")
    one_shot.add_argument('--input', help="annotate this jsonl/csv file")
    one_shot.add_argument('-o', '--output', help="where to annotate (default: <input>.annotated.jsonl)")
    one_shot.add_argument('--column', default='text', help="column holding the text")
    one_shot.add_argument('--label', default=None, help="column holding the gold label, if any")
    one_shot.add_argument('-n', '--limit', type=int, help="process only the first N rows")
    return argp


def drive(session, configs, args):
    """One run of one model: whichever of the four things `run_task` knows how to do. Everything
    that decides *which* model and *which* configuration got here lives in main()."""
    if args.interactive:
        TaskConsole(session).cmdloop(f"'{session.task_name}' on {args.model or 'the configured model'}: "
                                     f"type a text to run it, 'help -v' for the commands, "
                                     f"'model' to see what is loaded.")
    elif args.input:
        summary = session.annotate(args.input, output_path=args.output, column=args.column,
                                   label=args.label, limit=args.limit)
        print(f"{summary.pop('rows')} rows -> {summary.pop('output')}")
        print(format_summary(summary))
    elif args.text is not None:
        print(session.run_text(args.text))
    else:
        for name in configs.task.list_configs():
            session.use(name)
            print(f"\n=== {name} ({type(session.task_config).__name__}) ===")
            print(format_summary(session.run()))


def main():
    args = parser().parse_args()
    if args.attention:
        os.environ['MATFORMER_ATTENTION_NORMAL'] = args.attention
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    configs = MatformerConfigs.create_configs(args.cfg)
    overrides = list(args.override or [])
    if args.tokenizer:
        configs.import_config({"tokenizer": {"pretrained_name": args.tokenizer}})
    points = configs.sweep_points(args.sweep)
    if args.sweeps:
        sweeps.summarize(points)
        return
    if args.interactive and len(points) > 1:
        raise SystemExit("the console drives one model at a time: pick a point with --sweep "
                         f"({', '.join(point.name for point in points)})")
    # A sweep that touches the sections a model is built from needs one model per point; one that
    # sweeps tasks, data or hyperparameters instead shares the model and rebuilds only the run.
    rebuild, loaded = sweeps.crosses_model(configs), None
    for index, point in enumerate(points or [None], 1):
        if loaded is None or rebuild:
            # the first run, or a point that changes the model itself: build it from this point
            loaded = load_model(path=args.model, map_location=device, dtype=torch.bfloat16,
                                inference=True,
                                configs=configs if point is None else configs.sweep_variant(point))
            run = loaded['configs']           # what the checkpoint says, under what we say
        else:
            run = loaded['configs'].sweep_variant(point)
        if point is not None and len(points) > 1:
            print(f"\n=== Run {index}/{len(points)} of the sweep ===  {point.describe()}",
                  flush=True)
        model = loaded['model']
        session = TaskSession(model, run, task=args.task, overrides=overrides,
                              num_devices=args.num_devices,
                              source=args.model or run.checkpoint_path(),
                              presentation=args.present)
        untrained = f"untrained {type(model.model).__name__} built from the configs"
        print(f"Model: {args.model or run.checkpoint_path() or untrained}\n"
              f"Task: {session.task_name} ({type(session.task_config).__name__}) on {device}")
        drive(session, run, args)


if __name__ == '__main__':
    main()
