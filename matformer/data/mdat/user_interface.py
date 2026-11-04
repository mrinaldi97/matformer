"""MDAT command-line interface and interactive shell.

Run with ``python -m matformer.data.mdat.user_interface``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from .mdat import MatformerDataset

def fmt_bytes(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"


def parse_size(s: str) -> int:
    s = s.strip().upper()
    if not s:
        return 0
    unit = s[-1]
    if unit in "KMGTP":
        val = float(s[:-1])
        exponent = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[unit]
        return int(val * (1024**exponent))
    try:
        return int(float(s))
    except ValueError:
        return 0


def cmd_create(path: str, overwrite: bool, dsmap_bits: int, modalities: Sequence[str]):
    MatformerDataset.create_new(
        path, overwrite=overwrite, dsmap_bits=dsmap_bits, modalities=modalities
    )
    return f"Dataset created at {path}"


def cmd_load(
    path: str, readonly: bool = False, view: Optional[str] = None, shuffle: bool = False,
    strategy: Optional[str] = None, selection_json: Optional[str] = None,
):
    selection = None
    if selection_json is not None:
        with open(selection_json, encoding='utf-8') as stream:
            selection = json.load(stream)
    ds = MatformerDataset.load_dataset(
        path, readonly=readonly, ds_view=view, shuffle=shuffle,
        strategy=strategy, selection=selection,
    )
    return {
        "path": path,
        "ds_object": ds,
        "documents": ds.total_documents,
        "submdats": len(ds.list_submdat()),
        "strategies": len(ds.list_strategies()),
        "selection": ds.selection_summary,
    }


def cmd_add_submdat(
    path: str,
    name: str,
    data_compress: int = 9,
    meta_compress: int = 9,
    data_mapsize: str = "350G",
    meta_mapsize: str = "350G",
    modality: str = "text",
    ds: Optional[MatformerDataset] = None,
):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    ds.add_submdat(
        submdat_name=name,
        compression_levels={"data": data_compress, "meta": meta_compress},
        map_sizes={"data": parse_size(data_mapsize), "meta": parse_size(meta_mapsize)},
        modality=modality,
    )
    return f"Submdat '{name}' added"


def cmd_batch_import(
    path: str,
    folder: str,
    data_key: str = "text",
    data_compress: int = 9,
    meta_compress: int = 9,
    data_mapsize: str = "500G",
    meta_mapsize: str = "500G",
    pattern: str = "*.jsonl",
    recursive: bool = False,
    modality: str = "text",
    ds: Optional[MatformerDataset] = None,
):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    glob_path = os.path.join(folder, "**", pattern) if recursive else os.path.join(folder, pattern)
    files = glob.glob(glob_path, recursive=recursive)

    if not files:
        raise RuntimeError(f"No files matching '{glob_path}'")
    ok = 0
    for fp in files:
        name = Path(fp).stem
        print(f"Adding file {fp}...")
        sub = ds.add_submdat(
            name,
            compression_levels={"data": data_compress, "meta": meta_compress},
            map_sizes={
                "data": parse_size(data_mapsize),
                "meta": parse_size(meta_mapsize),
            },
            modality=modality,
        )
        res = sub.convert_to_submdat("jsonl", fp, data_key=data_key, progress_bar=True)
        if res:
            ok += 1
    return f"Batch import done: {ok}/{len(files)} succeeded"


def cmd_convert(path: str, submdat: str, _type: str, source: str, data_key: str = "text", ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    sub = ds[submdat]
    res = sub.convert_to_submdat(_type, source, data_key=data_key, progress_bar=True)
    if not res:
        raise RuntimeError("Conversion failed")
    return f"Converted {res.get('document_number', 0)} documents"


def cmd_list_submdats(path: str, detailed: bool = False, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    subs = ds.list_submdat()
    rows = []
    for name in subs:
        m = ds.db.get_manifest(submdat=name)
        rows.append(
            {
                "name": name,
                "docs": m.get("document_number", 0),
                "key": m.get("data_key"),
                "strategies": json.loads(m.get("pretokenization_strategies", "[]")),
            }
        )
    return rows


def cmd_list_strategies(path: str, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    return ds.list_strategies()


def cmd_register_strategy(path: str, name: str, json_path: str, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    with open(json_path) as f:
        ds.register_strategy(name, json.load(f))
    return f"Strategy '{name}' registered"


def cmd_pretokenize(
    path: str,
    submdat: str,
    strategy: str,
    parallel: bool = False,
    processes: Optional[int] = None,
    compression: int = 9,
    ds: Optional[MatformerDataset] = None,
):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    sub = ds.get_submdat(submdat)
    sub.pretokenize_submdat(
        strategy,
        progress_bar=True,
        parallel=parallel,
        num_processes=processes,
        compression_level=compression,
    )
    return "Pretokenization complete"

def cmd_pretokenize_batch(
    path: str,
    submdats: Sequence[str],
    strategy: str,
    parallel: bool = False,
    processes: Optional[int] = None,
    compression: int = 9,
    ds: Optional[MatformerDataset] = None,
):
    """Pretokenize multiple submdats with the same strategy."""
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)

    results = []
    failed = []

    for submdat_name in submdats:
        try:
            print(f"\nPretokenizing '{submdat_name}' with strategy '{strategy}'...")
            sub = ds.get_submdat(submdat_name)
            sub.pretokenize_submdat(
                strategy,
                progress_bar=True,
                parallel=parallel,
                num_processes=processes,
                compression_level=compression,
            )
            results.append(submdat_name)
            print(f"Completed: {submdat_name}")
        except Exception as e:
            failed.append((submdat_name, str(e)))
            print(f"Failed: {submdat_name} - {e}")

    summary = f"\nBatch pretokenization complete: {len(results)}/{len(submdats)} succeeded"
    if failed:
        summary += f"\nFailed ({len(failed)}):"
        for name, err in failed:
            summary += f"\n  - {name}: {err}"

    return summary

def cmd_export_view_template(path: str, output: str = "mock_view.json", ds: Optional[MatformerDataset] = None):

    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    submdats = ds.list_submdat()

    selection_dict = {name: "preserved" for name in submdats}

    tpl = {
        "view_name": "example_view",
        "selection": selection_dict,
        "shuffle": True
    }

    try:
        Path(output).write_text(json.dumps(tpl, indent=2))
        return f"View template written to {output}. Edit this file before creating the view."
    except Exception as e:
        return f"Error writing template: {e}"


def cmd_create_view(path: str, json_path: str, shuffle: bool = True, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    try:
        with open(json_path) as f:
            view_dict = json.load(f)
    except Exception as e:
        return f"Error reading JSON file {json_path}: {e}"

    if not shuffle:
        view_dict['shuffle'] = False

    if True:
        ds.register_new_view(view_dict)



        return f"View '{view_dict.get('view_name', 'N/A')}' created"



def cmd_list_views(path: str, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    views=ds.list_views()
    returns=[]
    for v in views:
        returns.append((v,ds.db.get_view_totals(v)['document_number']))

    return returns


def cmd_set_view(path: str, view: str, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    ds.set_view(view)
    return f"Active view set to '{view}'"

def cmd_export_view_data(
    path: str,
    view: str = "default",
    output: str = "export_view.jsonl",
    wanted: str = "document",
    with_meta: bool = True,
    limiter: Optional[int] = None,
    ds: Optional[MatformerDataset] = None
):
    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    ds.export_view(output_file=output, wanted=wanted, with_meta=with_meta, limiter=limiter, view=view)
    return f"View exported to '{output}'"

def cmd_shuffle_view(path: str, view: str, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=False)
    ds.shuffle(view)
    return f"View '{view}' shuffled"


def cmd_info(path: str, verbose: bool = False, ds: Optional[MatformerDataset] = None):
    ds = ds or MatformerDataset.load_dataset(path, readonly=True)
    info = {
        "path": path,
        "documents": ds.total_documents,
        "submdats": len(ds.list_submdat()),
        "strategies": len(ds.list_strategies()),
    }
    if verbose:
        info["manifest"] = ds.manifest
    return info


class InteractiveShell:
    """
    [REFACTOR]: Revamped TUI to hold the dataset object in memory,
    eliminating subprocess calls for massive performance gains and UI cleanup.
    """
    def __init__(self):
        self.dataset_path: Optional[str] = None
        self.ds: Optional[MatformerDataset] = None

    def _clear(self):
        """Clears the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def _pause(self):
        """Waits for user to press Enter."""
        try:
            input("\nPress Enter to continue...")
        except (KeyboardInterrupt, EOFError):
            pass

    def _call_cmd(self, func, *args, **kwargs):
        """Helper to call command functions, print results, and handle errors."""
        if True: #for debug
        #try:
            if self.ds:
                kwargs["ds"] = self.ds

            result = func(*args, **kwargs)
            print(f"\n {result}")
        #except Exception as e:
        #    print(f"\n Error: {e}")

    def prompt(self, text, default=None, cast=str):
        try:
            raw = input(
                f"{text}{f' [{default}]' if default is not None else ''}: "
            ).strip()
            if not raw and default is not None:
                return default
            if raw is None:
                return None
            return cast(raw)
        except (ValueError):
            print("Invalid input – try again")
            return self.prompt(text, default, cast)
        except (KeyboardInterrupt, EOFError):
            print()
            return None

    def pick(self, options, title="Select"):
        if not options:
            print("\nNo options to pick from.")
            return None

        print(f"\n{title}:")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        while True:
            try:
                idx_raw = self.prompt("Choice", cast=int)
                if idx_raw is None:
                    return None
                idx = int(idx_raw)
                if 1 <= idx <= len(options):
                    return options[idx - 1]
            except (ValueError):
                pass
            print("Out of range – try again")

    def main_loop(self):
        while True:
            self._clear()
            print("===  mdat interactive ===")
            if self.dataset_path:
                print(f"Current dataset: {self.dataset_path}")

            print("\n--- Main Menu ---")
            print("1. Create dataset")
            print("2. Load dataset")
            if self.dataset_path:
                print("3. Submdat operations")
                print("4. Strategy operations")
                print("5. View operations")
                print("6. Dataset info")
                print("7. Close dataset")
            print("0. Exit")

            choice_raw = self.prompt("Select", cast=int)
            if choice_raw is None: # Ctrl+C
                choice = 0
            else:
                choice = choice_raw

            if choice == 0:
                print("Exiting.")
                break

            pause = True
            if choice == 1:
                path = self.prompt("Path")
                if path:
                    overwrite = self.prompt("Overwrite (y/n)", "n").lower() == "y"
                    bits = self.prompt("dsmap bits", 8, cast=int)
                    self._call_cmd(
                        cmd_create,
                        path=path,
                        overwrite=overwrite,
                        dsmap_bits=bits,
                        modalities=["text"]
                    )
                else:
                    pause = False

            elif choice == 2:
                path = self.prompt("Path")
                if path:
                    try:
                        info = cmd_load(path, readonly=False)
                        self.ds = info["ds_object"]
                        self.dataset_path = path
                        print(f"\n Dataset loaded: {info['documents']:,} docs, "
                              f"{info['submdats']} submdats, {info['strategies']} strategies")
                    except Exception as e:
                        self.ds = None
                        self.dataset_path = None
                        print(f"\n Error loading dataset: {e}")
                else:
                    pause = False

            elif choice == 7:
                self.ds = None
                self.dataset_path = None
                print("\nDataset closed.")
                pause = True # Pause to show message

            elif self.dataset_path and choice == 6:
                try:
                    info = cmd_info(self.dataset_path, ds=self.ds)
                    print("\n--- Dataset Info ---")
                    print(f"Path: {info['path']}")
                    print(f"Documents: {info['documents']:,}")
                    print(f"Submdats: {info['submdats']}")
                    print(f"Strategies: {info['strategies']}")
                except Exception as e:
                    print(f"\n Error getting info: {e}")

            elif self.dataset_path and choice == 3:
                self.submdat_menu()
                pause = False

            elif self.dataset_path and choice == 4:
                self.strategy_menu()
                pause = False

            elif self.dataset_path and choice == 5:
                self.view_menu()
                pause = False

            elif choice > 7:
                print("Invalid choice.")

            if pause:
                self._pause()

    def submdat_menu(self):
        while True:
            self._clear()
            print(f"--- Submdat Operations [{self.dataset_path}] ---")
            print("1. List")
            print("2. Add")
            print("3. Batch import folder")
            print("4. Convert external data")
            print("5. Pretokenize (single)")
            print("6. Pretokenize (batch)")
            print("0. Back to Main Menu")

            c = self.prompt("Select", cast=int)
            if c is None or c == 0:
                return

            pause = True
            if c == 1:
                try:
                    rows = cmd_list_submdats(self.dataset_path, ds=self.ds)
                    if not rows:
                        print("\nNo submdats.")
                    else:
                        print("\n--- Submdats ---")
                        print(f"{'Name':<25} {'Docs':>10}  {'Key':<10}  Strategies")
                        print("-" * 60)
                        for r in rows:
                            print(
                                f"{r['name']:<25} {r['docs']:>10,}  {r['key'] or 'N/A':<10}  {','.join(r['strategies'])}"
                            )
                except Exception as e:
                    print(f"\n Error listing submdats: {e}")

            elif c == 2:
                name = self.prompt("Name")
                if name:
                    self._call_cmd(cmd_add_submdat, path=self.dataset_path, name=name)
                else:
                    pause = False

            elif c == 3:
                folder = self.prompt("Folder")
                if folder:
                    key = self.prompt("Data key", "text")
                    self._call_cmd(
                        cmd_batch_import,
                        path=self.dataset_path,
                        folder=folder,
                        data_key=key,
                    )
                else:
                    pause = False

            elif c == 4:
                subs = cmd_list_submdats(self.dataset_path, ds=self.ds)
                if not subs:
                    print("No submdats to convert into.")
                    pause = True
                    continue
                name = self.pick([r["name"] for r in subs], "Select Target Submdat")
                print(f"Selected {name}")
                if name:
                    tp = self.pick(["jsonl", "lmdb", "hf", "atlas"], "Select Source Type")
                    if tp:
                        src = self.prompt("Source path")
                        if src:
                            self._call_cmd(
                                cmd_convert,
                                path=self.dataset_path,
                                submdat=name,
                                _type=tp,
                                source=src,
                            )
                        else:
                            pause = False
                    else:
                        pause = False
                else:
                    pause = False

            elif c == 5:  # Single pretokenize
                subs = cmd_list_submdats(self.dataset_path, ds=self.ds)
                if not subs:
                    print("No submdats to pretokenize.")
                    pause = True
                    continue
                name = self.pick([r["name"] for r in subs], "Select Submdat")
                if name:
                    strategies = cmd_list_strategies(self.dataset_path, ds=self.ds)
                    if not strategies:
                        print("No strategies registered. Cannot pretokenize.")
                        pause = True
                        continue
                    st = self.pick(strategies, "Select Strategy")
                    if st:
                        self._call_cmd(
                            cmd_pretokenize,
                            path=self.dataset_path,
                            submdat=name,
                            strategy=st,
                            parallel=True,
                        )
                    else:
                        pause = False
                else:
                    pause = False

            elif c == 6:  # NEW: Batch pretokenize
                subs = cmd_list_submdats(self.dataset_path, ds=self.ds)
                if not subs:
                    print("No submdats to pretokenize.")
                    pause = True
                    continue

                # Get strategy first
                strategies = cmd_list_strategies(self.dataset_path, ds=self.ds)
                if not strategies:
                    print("No strategies registered. Cannot pretokenize.")
                    pause = True
                    continue

                st = self.pick(strategies, "Select Strategy")
                if not st:
                    pause = False
                    continue

                # Select submdats
                print("\n--- Select Submdats to Pretokenize ---")
                print("Options:")
                print("  1. All submdats")
                print("  2. Select multiple individually")

                choice = self.prompt("Choice", cast=int)

                selected_submdats = []
                if choice == 1:
                    selected_submdats = [r["name"] for r in subs]
                    print(f"\nSelected all {len(selected_submdats)} submdats")
                elif choice == 2:
                    print("\nEnter submdat names (comma-separated):")
                    available_names = [r["name"] for r in subs]
                    print(f"Available: {', '.join(available_names)}")

                    names_input = self.prompt("Submdats")
                    if names_input:
                        selected_submdats = [n.strip() for n in names_input.split(",")]
                        # Validate
                        invalid = [n for n in selected_submdats if n not in available_names]
                        if invalid:
                            print(f"Invalid submdat names: {', '.join(invalid)}")
                            selected_submdats = []
                else:
                    pause = False
                    continue

                if selected_submdats:
                    confirm = self.prompt(
                        f"Pretokenize {len(selected_submdats)} submdat(s) with '{st}'? (y/n)",
                        "y"
                    ).lower()
                    if confirm == "y":
                        self._call_cmd(
                            cmd_pretokenize_batch,
                            path=self.dataset_path,
                            submdats=selected_submdats,
                            strategy=st,
                            parallel=True,
                        )
                    else:
                        print("Cancelled.")
                else:
                    pause = False

            else:
                print("Invalid choice.")

            if pause:
                self._pause()

    def strategy_menu(self):
        while True:
            self._clear()
            print(f"--- Strategy Operations [{self.dataset_path}] ---")
            print("1. List")
            print("2. Register")
            print("0. Back to Main Menu")

            c = self.prompt("Select", cast=int)
            if c is None or c == 0:
                return

            pause = True
            if c == 1:
                try:
                    strategies = cmd_list_strategies(self.dataset_path, ds=self.ds)
                    if not strategies:
                        print("\nNo strategies registered.")
                    else:
                        print("\n--- Registered Strategies ---")
                        print("\n".join(f"- {s}" for s in strategies))
                except Exception as e:
                    print(f"\n Error listing strategies: {e}")

            elif c == 2:
                name = self.prompt("Strategy name")
                if name:
                    json_path = self.prompt("JSON file")
                    if json_path:
                        self._call_cmd(
                            cmd_register_strategy,
                            path=self.dataset_path,
                            name=name,
                            json_path=json_path,
                        )
                    else:
                        pause = False
                else:
                    pause = False
            else:
                print("Invalid choice.")

            if pause:
                self._pause()
    def _pick_view(self, prompt_msg="Select a view"):
            views = [v[0] for v in cmd_list_views(self.dataset_path, ds=self.ds)]
            if not views:
                print("\nNo views available.")
                return None
            return self.pick(views, prompt_msg)
    def view_menu(self):

        while True:
            self._clear()
            print(f"--- View Operations [{self.dataset_path}] ---")
            print("1. List")
            print("2. Export mock view (JSON)")  # <-- UPDATED TEXT
            print("3. Create from JSON")
            print("4. Set active")
            print("5. Shuffle")
            print("6. Export data to file")
            print("0. Back to Main Menu")

            c = self.prompt("Select", cast=int)
            if c is None or c == 0:
                return

            pause = True
            if c == 1:
                if True:
                    views = cmd_list_views(self.dataset_path, ds=self.ds)
                    if not views:
                        print("\nNo views.")
                    else:
                        print("\n--- Views ---")
                        print(f"{'Name':<30} {'Docs':<10}")
                        print("-" * 40)
                        for name, docs in views:
                            doc_str = f"{docs:<10,}" if docs is not None else "N/A"
                            print(f"{name:<30} {doc_str}")
                #except Exception as e:
                #    print(f"\n Error listing views: {e}")

            elif c == 2:
                out = self.prompt("Output file", "mock_view.json")
                if out:
                    self._call_cmd(cmd_export_view_template, path=self.dataset_path, output=out)
                    print(f"Edit {out} then use option 3")
                else:
                    pause = False

            elif c == 3:
                json_path = self.prompt("JSON file")
                if json_path:
                    self._call_cmd(cmd_create_view, path=self.dataset_path, json_path=json_path)
                else:
                    pause = False

            elif c in [4, 5]:  # Set or Shuffle
                v = self._pick_view(f"Select View to {'set' if c==4 else 'shuffle'}")
                if v:
                    func = cmd_set_view if c == 4 else cmd_shuffle_view
                    self._call_cmd(func, path=self.dataset_path, view=v)
                else:
                    pause = False

            elif c == 6:
                v = self._pick_view("Select View to export")
                if not v:
                    pause = False
                    continue

                out = self.prompt("Output file", "export_view.jsonl")
                if not out:
                    pause = False
                    continue

                wanted = self.pick(["document", "tokens", "chunked_tokens"], "Select item type")
                with_meta = self.prompt("Include metadata? (y/n)", "y").lower() == "y"
                limiter_raw = self.prompt("Limiter (max items, empty for no limit)", "")
                limiter = int(limiter_raw) if limiter_raw.strip() else None

                self._call_cmd(
                    cmd_export_view_data,
                    path=self.dataset_path,
                    view=v,
                    output=out,
                    wanted=wanted,
                    with_meta=with_meta,
                    limiter=limiter
                )
                print(f"View '{v}' exported to {out}")


            else:
                print("Invalid choice.")

            if pause:
                self._pause()


def build_cli():
    p = argparse.ArgumentParser(
        description="mdat – manage Matformer datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tui", action="store_true", help="Start interactive TUI mode")
    sp = p.add_subparsers(dest="cmd", required=False, help="Sub-command to run")

    path_parser = argparse.ArgumentParser(add_help=False)
    path_parser.add_argument("--path", required=True, help="Path to the dataset")

    submdat_parser = argparse.ArgumentParser(add_help=False)
    submdat_parser.add_argument(
        "--data-compress", type=int, default=0, help="Data compression level"
    )
    submdat_parser.add_argument(
        "--meta-compress", type=int, default=0, help="Metadata compression level"
    )
    submdat_parser.add_argument(
        "--data-mapsize", default="10G", help="LMDB map size for data"
    )
    submdat_parser.add_argument(
        "--meta-mapsize", default="1G", help="LMDB map size for metadata"
    )

    # --- Command Definitions ---
    p_create = sp.add_parser(
        "create", parents=[path_parser], help="Create a new dataset"
    )
    p_create.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing dataset"
    )
    p_create.add_argument("--dsmap-bits", type=int, default=8)
    p_create.add_argument("--modalities", nargs="+", default=["text"])

    p_load = sp.add_parser("load", parents=[path_parser], help="Load a dataset")
    p_load.add_argument("--readonly", action="store_true")
    p_load.add_argument("--view", help="Load a specific view")
    p_load.add_argument("--shuffle", action="store_true", help="Shuffle view on load")
    p_load.add_argument("--strategy", help="Pretokenization strategy used to count tokens")
    p_load.add_argument("--selection-json", help="JSON file with domains, max_tokens, domain_tokens, and seed")

    p_add_sub = sp.add_parser(
        "add-submdat",
        parents=[path_parser, submdat_parser],
        help="Add a new, empty submdat",
    )
    p_add_sub.add_argument("--name", required=True, help="Name of the new submdat")
    p_add_sub.add_argument("--modality", default="text")

    p_batch = sp.add_parser(
        "batch-import",
        parents=[path_parser, submdat_parser],
        help="Import a folder of files",
    )
    p_batch.add_argument("--folder", required=True, help="Folder to import from")
    p_batch.add_argument("--data-key", default="text", help="JSON key for data")
    p_batch.add_argument("--pattern", default="*.jsonl", help="File pattern to match")
    p_batch.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    p_batch.add_argument("--modality", default="text")

    p_conv = sp.add_parser(
        "convert", parents=[path_parser], help="Convert external data into a submdat"
    )
    p_conv.add_argument("--submdat", required=True, help="Target submdat name")
    p_conv.add_argument(
        "--type",
        dest="_type",
        required=True,
        choices=["jsonl", "lmdb", "hf", "atlas"],
        help="Source data type",
    )
    p_conv.add_argument("--source", required=True, help="Source file or path")
    p_conv.add_argument("--data-key", default="text")

    p_ls_sub = sp.add_parser(
        "list-submdats", parents=[path_parser], help="List submdats in the dataset"
    )
    p_ls_sub.add_argument("--detailed", action="store_true", help="Show more details")

    sp.add_parser(
        "list-strategies", parents=[path_parser], help="List registered strategies"
    )

    p_reg_strat = sp.add_parser(
        "register-strategy",
        parents=[path_parser],
        help="Register a new strategy from JSON",
    )
    p_reg_strat.add_argument("--name", required=True, help="Name for the strategy")
    p_reg_strat.add_argument("--json", dest="json_path", required=True, help="Path to the strategy JSON file")

    p_pretok = sp.add_parser(
        "pretokenize", parents=[path_parser], help="Pretokenize a submdat"
    )
    p_pretok.add_argument("--submdat", required=True, help="Submdat to pretokenize")
    p_pretok.add_argument("--strategy", required=True, help="Strategy to use")
    p_pretok.add_argument(
        "--parallel", action="store_true", help="Use parallel processing"
    )
    p_pretok.add_argument("--processes", type=int, help="Number of processes to use")
    p_pretok.add_argument("--compression", type=int, default=0)

    p_export_view = sp.add_parser(
        "export-view",
        parents=[path_parser],
        help="Export a mock view JSON (all submdats 'preserved')"  # <-- UPDATED HELP
    )
    p_export_view.add_argument(
        "--output",
        default="mock_view.json",  # <-- UPDATED DEFAULT
        help="Output file name"
    )
    p_export_view_data = sp.add_parser(
        "export-view-stream",
        parents=[path_parser],
        help="Stream export the active view to JSONL (TB-scale, selectable metadata, type, and limiter)"
    )
    p_export_view_data.add_argument("--output", default="export_view.jsonl", help="Output file path")
    p_export_view_data.add_argument(
        "--wanted", default="document", choices=["document", "default"], help="Type of items to export"
    )
    p_export_view_data.add_argument(
        "--view", type=str,default="default", help="View to export"
    )
    p_export_view_data.add_argument("--no-meta", dest="with_meta", action="store_false",
                                   help="Exclude metadata from export")
    p_export_view_data.add_argument(
        "--limiter", type=int, default=None, help="Maximum number of items to export"
    )
    p_create_view = sp.add_parser(
        "create-view", parents=[path_parser], help="Create a view from JSON"
    )
    p_create_view.add_argument("--json", dest="json_path", required=True, help="Path to the view JSON file")
    p_create_view.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Do not shuffle the view (overrides 'shuffle' key in JSON)",  # <-- UPDATED HELP
    )

    sp.add_parser("list-views", parents=[path_parser], help="List all views")

    p_set_view = sp.add_parser(
        "set-view", parents=[path_parser], help="Set the active view"
    )
    p_set_view.add_argument("--view", required=True, help="View name to activate")

    p_shuffle_view = sp.add_parser(
        "shuffle-view", parents=[path_parser], help="Shuffle a view"
    )
    p_shuffle_view.add_argument("--view", required=True, help="View name to shuffle")


    p_info = sp.add_parser(
        "info", parents=[path_parser], help="Show dataset info"
    )
    p_info.add_argument("--verbose", action="store_true", help="Show full manifest")
    #Batch pretokenize
    p_pretok_batch = sp.add_parser(
        "pretokenize-batch",
        parents=[path_parser],
        help="Pretokenize multiple submdats with the same strategy"
    )
    p_pretok_batch.add_argument(
        "--submdats",
        nargs="+",
        help="Submdat names to pretokenize (space-separated)"
    )
    p_pretok_batch.add_argument("--strategy", required=True, help="Strategy to use")
    p_pretok_batch.add_argument(
        "--parallel", action="store_true", help="Use parallel processing"
    )
    p_pretok_batch.add_argument("--processes", type=int, help="Number of processes to use")
    p_pretok_batch.add_argument("--compression", type=int, default=0)
    p_pretok_batch.add_argument(
        "--all",
        action="store_true",
        help="Pretokenize all submdats (ignores --submdats)"
    )
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if not argv or (argv and argv[0] == "--tui"):
        InteractiveShell().main_loop()
        return

    parser = build_cli()
    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return

    cmd_func_map = {
        "create": cmd_create,
        "load": cmd_load,
        "add-submdat": cmd_add_submdat,
        "batch-import": cmd_batch_import,
        "convert": cmd_convert,
        "list-submdats": cmd_list_submdats,
        "list-strategies": cmd_list_strategies,
        "register-strategy": cmd_register_strategy,
        "pretokenize": cmd_pretokenize,
        "pretokenize-batch": cmd_pretokenize_batch,
        "export-view": cmd_export_view_template,
        "export-view-stream": cmd_export_view_data,
        "create-view": cmd_create_view,
        "list-views": cmd_list_views,
        "set-view": cmd_set_view,
        "shuffle-view": cmd_shuffle_view,
        "info": cmd_info,
    }

    try:
        kwargs = vars(args)
        cmd_name = kwargs.pop("cmd")
        kwargs.pop("tui", None)

        cmd_func = cmd_func_map.get(cmd_name)
        if not cmd_func:
            raise NotImplementedError(f"Command '{cmd_name}' not implemented in main")

        if cmd_name == "pretokenize-batch":
            all_domains = kwargs.pop("all", False)
            if all_domains:
                with MatformerDataset.load_dataset(kwargs["path"], readonly=True) as dataset:
                    kwargs["submdats"] = dataset.list_submdat()
            elif not kwargs.get("submdats"):
                parser.error("pretokenize-batch requires --submdats or --all")
        result = cmd_func(**kwargs)

        # --- Handle printing results for CLI ---
        if cmd_name == "load":
            info = result
            print(
                f"Dataset loaded: {info['documents']:,} docs, "
                f"{info['submdats']} submdats, {info['strategies']} strategies"
            )
            if info['selection'] is not None:
                print(json.dumps(info['selection'], indent=2))
            info['ds_object'].close()
        elif cmd_name == "list-submdats":
            rows = result
            if not rows:
                print("No submdats")
            elif args.detailed:
                for r in rows:
                    print(
                        f"{r['name']:<25} – {r['docs']:>10,} docs – "
                        f"key={r['key']} – strategies={','.join(r['strategies'])}"
                    )
            else:
                print(f"{'Name':<30} {'Docs':<10}")
                print("-" * 40)
                for r in rows:
                    print(f"{r['name']:<30} {r['docs']:<10,}")
        elif cmd_name == "list-strategies":
            strategies = result
            print("\n".join(strategies) if strategies else "No strategies")
        elif cmd_name == "list-views":
            views = result
            if not views:
                print("No views")
            else:
                print(f"{'Name':<30} {'Docs':<10}")
                print("-" * 40)
                for name, docs in views:
                    print(f"{name:<30} {docs:<10,}")
        elif cmd_name == "info":
            info = result
            print(f"Path: {info['path']}")
            print(f"Documents: {info['documents']:,}")
            print(f"Submdats: {info['submdats']}")
            print(f"Strategies: {info['strategies']}")
            if args.verbose and "manifest" in info:
                print("Manifest:", json.dumps(info["manifest"], indent=2))
        elif result is not None:
            print(result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
