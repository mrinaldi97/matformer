import argparse, torch, sys, re
from matformer.models import PL_ModelWrapper
from matformer.tokenizers import MatformerTokenizer
from transformers import AutoTokenizer
from matformer.transformer_blocks import Autoregressive_Model, BERTModel, EntropyModel

# ---- ARGS ----
p = argparse.ArgumentParser("Matformer inference")
p.add_argument('--model', required=True)
p.add_argument('--arch', required=True,
               choices=['gpt', 'bert', 'entropy'],
               help='Pick architecture: gpt=autoregressive, bert=masked, entropy=entropy model')
p.add_argument('--mode', default=None,
               choices=['gen', 'entropy'],
               help='For entropy model: gen=generate like GPT, entropy=entropy analysis')
p.add_argument('--prompt', default=None)
p.add_argument('--length', type=int, default=100)
p.add_argument('--temp', type=float, default=0.6)
p.add_argument('--top_k', type=int, default=0)
p.add_argument('--top_p', type=float, default=0.9)
p.add_argument('--interactive', action='store_true')
p.add_argument('--tokenizer', default=None)
p.add_argument('--masking_ratio', type=float, default=0.25)
p.add_argument('--smoothing', type=float, default=None)
args = p.parse_args()

# ---- MAP ARCH ----
if args.arch == 'gpt':
    ModelClass = Autoregressive_Model
elif args.arch == 'bert':
    ModelClass = BERTModel
elif args.arch == 'entropy':
    ModelClass = EntropyModel
else:
    print("Unknown arch")
    sys.exit(1)

# ---- LOAD ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok_arg = 'bytes' if args.tokenizer is None else args.tokenizer
model, cfg = PL_ModelWrapper.load_from_checkpoint(
    checkpoint_path=args.model,
    ModelClass=ModelClass,
    map_location=device,
    inference_fix=True,
    tokenizer=tok_arg
)
print("Loaded model on", device)
print("Config:", cfg)
model = model.to(device).eval()

# ---- FNS ----
def do_autoreg(prompt):
    out = model.model.generate(prompt=prompt, max_length=args.length,
                               temperature=args.temp, top_k=args.top_k, top_p=args.top_p)
    print("\n--- Generated ---\n" + out + "\n------\n")

def do_masked(prompt):
    acc, toks = model.model.inference_testing(prompt, masking_ratio=args.masking_ratio)
    print("\n--- Masked prediction ---")
    print(" ".join(toks))
    print(f"Accuracy: {acc*100:.2f}%\n------\n")

def do_entropy(prompt):
    ent = model.model.compute_entropy(prompt)
    print("\n--- Entropy per token ---")
    print(ent.squeeze().tolist())
    if args.smoothing is not None:
        cuts, cmask, gmask = model.model.monotonicity_breakpoints(prompt=prompt, smoothing=args.smoothing)
        print("Cutting points:", cuts)
        try:
            chunks = model.model.cut_text(prompt, cutting_points=cuts[0])
            print("Chunks:", chunks)
        except Exception as e:
            print("Cut text failed:", e)
    print("------\n")

def run_once(prompt):
    if args.arch == 'bert':
        do_masked(prompt)
    elif args.arch == 'entropy':
        if args.mode == 'gen':
            do_autoreg(prompt)
        else:
            do_entropy(prompt)
    else:
        do_autoreg(prompt)

# ---- MAIN ----
if args.interactive:
    print("Interactive mode. Ctrl+C to quit.")
    while True:
        try:
            line = input(">> ").strip()
            if not line:
                continue
            if line.startswith("/mode "):
                new_mode = line.split(maxsplit=1)[1].strip()
                if new_mode in ["gen", "entropy"]:
                    args.mode = new_mode
                    print(f"Mode switched to: {args.mode}")
                else:
                    print("Invalid mode. Use 'gen' or 'entropy'.")
                continue
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
