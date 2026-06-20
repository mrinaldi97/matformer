"""
Checkpoint parameter mapping utility with automatic layer expansion.

Usage:
    # 1. Extract shapes from checkpoint
    python checkpoint_mapper.py dump old_ckpt.pt old_shapes.json
    
    # 2. Generate editable CSV (optionally with new model shapes for auto-matching)
    python checkpoint_mapper.py template old_shapes.json [new_shapes.json] -o mapping.csv
    
    # 3. Edit mapping.csv, then export final JSON with auto-expanded layers
    python checkpoint_mapper.py export mapping.csv -o final_mapping.json
"""

import json, csv, re, argparse
from pathlib import Path

def load_checkpoint(path):
    """Load checkpoint and return state_dict (lazy imports torch)."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("state_dict", ckpt)

def get_shapes(state_dict):
    """Extract parameter shapes as strings."""
    return {k: "x".join(map(str, v.shape)) for k, v in state_dict.items()}

def normalize_layers(name, layer_count=None):
    """Convert layer indices to .Nn. placeholder where n is max layer count."""
    if layer_count is None:
        return re.sub(r'\.layers\.(\d+)\.', '.layers.N.', name)
    return re.sub(r'\.layers\.(\d+)\.', f'.layers.N{layer_count}.', name)

def find_layer_counts(names):
    """Detect max layer index for each layer pattern."""
    counts = {}
    for name in names:
        if m := re.search(r'(.*\.layers\.)(\d+)(\..*)', name):
            pattern = f"{m.group(1)}N{m.group(3)}"
            counts[pattern] = max(counts.get(pattern, -1), int(m.group(2)))
    return counts

def expand_layers(mapping):
    """Expand .Nn. placeholders to actual layer indices using embedded count."""
    expanded = {}
    for stable, target in mapping.items():
        if m := re.search(r'\.layers\.N(\d+)\.', stable):
            max_idx = int(m.group(1))
            for i in range(max_idx + 1):
                expanded[re.sub(r'\.layers\.N\d+\.', f'.layers.{i}.', stable)] = target
        else:
            expanded[stable] = target
    return expanded

def cmd_dump(args):
    """Extract checkpoint shapes to JSON."""
    state_dict = load_checkpoint(args.checkpoint)
    shapes = get_shapes(state_dict)
    Path(args.output).write_text(json.dumps(shapes, indent=2))
    print(f"[OK] Saved {len(shapes)} parameter shapes to {args.output}")

def cmd_template(args):
    """Generate CSV mapping template with optional auto-matching."""
    old_shapes = json.loads(Path(args.old_json).read_text())
    new_shapes = json.loads(Path(args.new_json).read_text()) if args.new_json else {}
    
    # Detect layer counts and collapse old parameters into stable names
    layer_counts = find_layer_counts(old_shapes.keys())
    stable_to_shape = {}
    for old_name, shape in old_shapes.items():
        stable = normalize_layers(old_name)
        if stable not in stable_to_shape:
            # Embed layer count in the name
            count = layer_counts.get(stable, 0)
            stable_with_count = normalize_layers(old_name, count)
            stable_to_shape[stable_with_count] = shape
    
    # Generate CSV with collapsed names
    rows = []
    for stable, shape in stable_to_shape.items():
        # Try auto-matching by shape
        candidates = [k for k, v in new_shapes.items() if v == shape] if new_shapes else []
        new_name = candidates[0] if len(candidates) == 1 else ""
        rows.append([stable, new_name])
    
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["old_name", "new_name"])
        writer.writerows(rows)
    
    matched = sum(1 for _, new in rows if new)
    print(f"[OK] Created template with {matched}/{len(rows)} auto-matched parameters")
    print(f"     Edit {args.output} to complete mappings, then run 'export' command")

def cmd_export(args):
    """Export final mapping with auto-expanded layers."""
    # Read CSV mapping (names contain embedded layer counts as .Nn.)
    mapping = {}
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            old, new = row["old_name"], row["new_name"] or row["old_name"]
            mapping[old] = new
    
    # Expand all .Nn. patterns using embedded counts
    final = expand_layers(mapping)
    
    Path(args.output).write_text(json.dumps(final, indent=2))
    print(f"[OK] Exported {len(final)} mappings (expanded from {len(mapping)}) to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # dump command
    p = subparsers.add_parser("dump", help="Extract checkpoint shapes")
    p.add_argument("checkpoint", help="Path to checkpoint or safetensor file")
    p.add_argument("output", help="Output JSON path")
    
    # template command
    p = subparsers.add_parser("template", help="Generate CSV mapping template")
    p.add_argument("old_json", help="Old checkpoint shapes JSON")
    p.add_argument("new_json", nargs="?", help="New model shapes JSON (optional, for auto-matching)")
    p.add_argument("-o", "--output", default="mapping.csv", help="Output CSV path")
    
    # export command
    p = subparsers.add_parser("export", help="Export final mapping with expanded layers")
    p.add_argument("csv", help="Edited mapping CSV with .Nn. notation")
    p.add_argument("-o", "--output", default="external_mapping.json", help="Output mapping JSON")
    
    args = parser.parse_args()
    {"dump": cmd_dump, "template": cmd_template, "export": cmd_export}[args.command](args)
