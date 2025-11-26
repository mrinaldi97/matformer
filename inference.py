#inference.py
import argparse, torch, sys, re, os
from matformer.models import PL_ModelWrapper
from matformer.matformer_tokenizers import MatformerTokenizer
from transformers import AutoTokenizer
from matformer.transformer_blocks import Autoregressive_Model, BERTModel, EntropyModel
from copy import deepcopy
from statistics import mean


# ---- LOAD ----
def load_inference_model(checkpoint_path, ModelClass, map_location, tokenizer):
    if ModelClass==BERTModel:
        overrides={'is_causal':False}
    elif ModelClass==Autoregressive_Model:
        overrides={'is_causal':True}
    model, cfg = PL_ModelWrapper.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        ModelClass=ModelClass,
        map_location=map_location,
        tokenizer=tokenizer,
        overrides=overrides
    )

    model = model.to(map_location).to(torch.bfloat16).eval()
    for module in model.modules():
        if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
            module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)
    return model, cfg


def compute_entropy(model, prompt, return_type='chunks', smoothing=0.0, hard_limit=None):
    ent = model.model.compute_entropy(prompt)
    cuts, cmask, gmask = model.model.monotonicity_breakpoints(prompt=prompt, smoothing=smoothing)
    chunks = [x for x in model.model.cut_text(prompt, cutting_points=cuts, hard_limit=hard_limit) if len(x) > 0]
    if return_type == 'dict':
        return {"chunks": chunks, "cuts": cuts, "cmask": cmask, "gmask": gmask, "ent": ent}
    elif return_type == 'chunks':
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
    p.add_argument('--txt_file', default=None, help='Path to .txt file for BERT testing')
    p.add_argument('--chunk_size', type=int, default=1024, help='Chunk size in tokens for .txt inference')
    p.add_argument('--report', action='store_true', help='Generate detailed report for .txt input')
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

    model, cfg = load_inference_model(checkpoint_path=args.model, ModelClass=ModelClass, map_location=device, tokenizer=tok_arg)
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
        return acc, toks

    def do_entropy(prompt):
        return_dict = compute_entropy(model=model, prompt=prompt, return_type='dict', smoothing=args.smoothing)
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

    # ---- TXT FILE INFERENCE ----
    if args.txt_file is not None:
        if args.arch != 'bert':
            print("Error: --txt_file can only be used with --arch bert.")
            sys.exit(1)

        if not os.path.exists(args.txt_file):
            print(f"File not found: {args.txt_file}")
            sys.exit(1)

        with open(args.txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        print(f"Loaded text file: {args.txt_file}")
        tokens = model.model.tokenizer.encode(text)
        total_tokens = len(tokens)
        chunk_size = args.chunk_size
        num_chunks = (total_tokens + chunk_size - 1) // chunk_size
        print(f"Total tokens: {total_tokens}, divided into {num_chunks} chunks of {chunk_size} tokens")

        results = []
        for i in range(num_chunks):
            chunk_tokens = tokens[i*chunk_size:(i+1)*chunk_size]
            decoded_chunk = model.model.tokenizer.decode(chunk_tokens)
            acc, out_toks = model.model.inference_testing(input_text=None, masking_ratio=args.masking_ratio, tokens=chunk_tokens)
            results.append({
                "index": i,
                "accuracy": acc,
                "decoded_chunk": decoded_chunk,
                "predicted_output": " ".join(out_toks)
            })
            print(f"Chunk {i+1}/{num_chunks} | Accuracy: {acc*100:.2f}%")

        accuracies = [r["accuracy"] for r in results]
        avg_acc = mean(accuracies)
        best_chunk = max(results, key=lambda x: x["accuracy"])
        worst_chunk = min(results, key=lambda x: x["accuracy"])

        print(f"\nOverall average accuracy: {avg_acc*100:.2f}%")
        #print(f"Best chunk #{best_chunk['index']} accuracy: {best_chunk['accuracy']*100:.2f}%")
        #print(f"Worst chunk #{worst_chunk['index']} accuracy: {worst_chunk['accuracy']*100:.2f}%")

        if args.report:
            base_name = os.path.splitext(os.path.basename(args.txt_file))[0]
            report_name = f"{base_name}_record.txt"
            with open(report_name, 'w', encoding='utf-8') as rep:
                rep.write(f"File: {args.txt_file}\n")
                rep.write(f"Total tokens: {total_tokens}\n")
                rep.write(f"Chunks: {num_chunks}\n")
                rep.write(f"Average accuracy: {avg_acc*100:.2f}%\n")
                rep.write(f"Best chunk index: {best_chunk['index']} ({best_chunk['accuracy']*100:.2f}%)\n")
                rep.write(f"Worst chunk index: {worst_chunk['index']} ({worst_chunk['accuracy']*100:.2f}%)\n\n")

                rep.write("---- BEST CHUNK ----\n")
                rep.write(best_chunk["decoded_chunk"] + "\n\nPredicted:\n" + best_chunk["predicted_output"] + "\n\n")
                rep.write("---- WORST CHUNK ----\n")
                rep.write(worst_chunk["decoded_chunk"] + "\n\nPredicted:\n" + worst_chunk["predicted_output"] + "\n\n")

                rep.write("---- ALL CHUNK ACCURACIES ----\n")
                for r in results:
                    rep.write(f"Chunk {r['index']}: {r['accuracy']*100:.2f}%\n")
                    

            print(f"Report written to {report_name}")

        sys.exit(0)

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
        if args.prompt is None and args.txt_file is None:
            print("Need --prompt or --txt_file if not in interactive mode.")
            sys.exit(1)
        if args.prompt is not None:
            run_once(args.prompt)
