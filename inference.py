import argparse, torch, sys, re
from matformer.models import PL_ModelWrapper
from matformer.matformer_tokenizers import MatformerTokenizer
from transformers import AutoTokenizer
from matformer.transformer_blocks import Autoregressive_Model, BERTModel, EntropyModel


# ---- LOAD ----
def load_inference_model(checkpoint_path,ModelClass,map_location,tokenizer):
    model, cfg = PL_ModelWrapper.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        ModelClass=ModelClass,
        map_location=map_location,
        tokenizer=tokenizer
    )

    model = model.to(map_location).to(torch.bfloat16).eval()
    for module in model.modules():
        if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
            module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)
    return model,cfg
            
def compute_entropy(model,prompt,return_type='chunks',smoothing=0.0,hard_limit=None):
    #print(f"DEBUG: {prompt}")
    ent=model.model.compute_entropy(prompt)
    cuts, cmask, gmask = model.model.monotonicity_breakpoints(prompt=prompt, smoothing=smoothing)
    #print(f"DEBUG: {cuts}")
    chunks = [x for x in model.model.cut_text(prompt, cutting_points=cuts,hard_limit=hard_limit) if len(x)>0]

    if return_type=='dict':
        return {"chunks":chunks,"cuts":cuts,"cmask":cmask,"gmask":gmask,"ent":ent}
    elif return_type=='chunks':
        return chunks
    else:
        return None 
        
        
if __name__ == "__main__":             
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok_arg = 'bytes' if args.tokenizer is None else args.tokenizer


    model,cfg=load_inference_model(checkpoint_path=args.model,ModelClass=ModelClass,map_location=device,tokenizer=tok_arg)
    print("Loaded model on", device)
    print("Config:", cfg)
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
        return_dict=compute_entropy(model=model,prompt=prompt,return_type='dict',smoothing=args.smoothing)
        
        print("Chunks:", return_dict['chunks'])
        print(f"Text divided into {len(return_dict['chunks'])} chunks")
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
