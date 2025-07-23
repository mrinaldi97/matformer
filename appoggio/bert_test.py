import argparse
import torch
from matformer.tokenizers import MatformerTokenizer
from matformer.model_config import ModelConfig
from matformer.models import BERTModel

parser = argparse.ArgumentParser(description='BERT inference test')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--interactive', action='store_true')
parser.add_argument('--masking_ratio', type=float, default=0.25)
parser.add_argument('--tokenizer',type=str,default="sapienzanlp/Minerva-350M-base-v1.0")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, config = BERTModel.load_from_checkpoint(args.model, inference_fix=True, map_location=device, tokenizer=args.tokenizer)
model = model.to(device, torch.bfloat16)
model.eval()

def show_output(input_text):
    acc, out_tokens = model.inference_testing(input_text, masking_ratio=args.masking_ratio)
    masked_str = input_text
    print(f"\nInput: {input_text}")
    print(f"\nOutput: {out_tokens}")

    print(f"\nAccuracy: {acc:.2f}")

if args.interactive:
    while True:
        try:
            inp = input(">>> ")
            if not inp:
                continue
            show_output(inp)
        except KeyboardInterrupt:
            print("\nBye.")
            break

elif args.prompt:
    show_output(args.prompt)

elif args.input and args.output:
    with open(args.input, "r") as f:
        lines = f.readlines()
    with open(args.output, "w") as f:
        for line in lines:
            line = line.strip()
            acc, predicted = model.inference_testing(line, masking_ratio=args.masking_ratio)
            out = " ".join(f"[{tok}]" for tok in predicted)
            f.write(f"{line}\n{out}\n\n")
    print(f"Saved to {args.output}")

else:
    print("Provide --interactive OR --prompt OR both --input and --output")
