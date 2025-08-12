import argparse, torch, sys
from matformer.models import PL_ModelWrapper
from matformer.tokenizers import MatformerTokenizer
from transformers import AutoTokenizer
import re

# ---- ARGS ----
p = argparse.ArgumentParser("Matformer inference")
p.add_argument('--model', required=True)
p.add_argument('--prompt', default=None)
p.add_argument('--length', type=int, default=100)
p.add_argument('--temp', type=float, default=0.6)
p.add_argument('--top_k', type=int, default=0)
p.add_argument('--top_p', type=float, default=0.9)
p.add_argument('--interactive', action='store_true')
p.add_argument('--tokenizer', default=None)
p.add_argument('--masking_ratio', type=float, default=0.25)   # for BERT-like
p.add_argument('--smoothing', type=float, default=None)       # for entropy
args = p.parse_args()

# ---- LOAD ----
ckpt = args.model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, cfg = PL_ModelWrapper.load_from_checkpoint(ckpt, map_location=device, inference_fix=True)
print("Loaded model on", device)
print("Config:", cfg)
model = model.to(device).eval()
tok = 'bytes' if args.tokenizer is None else AutoTokenizer.from_pretrained(args.tokenizer)
tok = MatformerTokenizer(cfg, tokenizer=tok)

# ---- INTERACTIVE LOOP ----
def do_autoreg(prompt):
    out = model.generate(prompt=prompt, max_length=args.length,
                         temperature=args.temp, top_k=args.top_k, top_p=args.top_p)
    print("\n--- Generated ---")
    print(out)
    print("------\n")

def do_masked(prompt):
    try:
        acc, toks = model.model.inference_testing(prompt, masking_ratio=args.masking_ratio)
        print("\n--- Masked prediction ---")
        print(" ".join(toks))
        print(f"Accuracy: {acc*100:.2f}%")
        print("------\n")
    except AttributeError:
        print("This model does not support masked inference.")

def do_entropy(prompt):
    ent = model.model.compute_entropy(prompt)
    print("\n--- Entropy per token ---")
    print(ent.squeeze().tolist())
    if args.smoothing is not None:
        cuts, cmask, gmask = model.model.monotonicity_breakpoints(prompt=prompt, smoothing=args.smoothing)
        print("Cutting points:", cuts)
        txt = prompt
        try:
            chunks = model.model.cut_text(txt, cutting_points=cuts[0])
            print("Chunks:", chunks)
        except Exception as e:
            print("Cut text failed:", e)
    print("------\n")

def run_once(prompt):
    t = cfg.get("training_objective", None)
    if 'masked' in str(t).lower():
        do_masked(prompt)
    elif 'entropy' in str(type(model.model)).lower():
        do_entropy(prompt)
    else:
        do_autoreg(prompt)

if args.interactive:
    print("Interactive mode. Ctrl+C to quit.")
    while True:
        try:
            line = input(">> ").strip()
            if not line: continue
            # [CUT x.x] syntax for entropy smoothing
            m = re.match(r"\[CUT ([0-9.]+)\](.*)", line)
            if m:
                args.smoothing = float(m.group(1))
                line = m.group(2).strip()
                do_entropy(line)
            else:
                run_once(line)
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
else:
    if args.prompt is None:
        print("Need --prompt if not in interactive mode.")
        sys.exit(1)
    run_once(args.prompt)
