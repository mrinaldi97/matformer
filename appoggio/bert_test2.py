import argparse
from matformer.models import BERTModel
from inference.inference_code import setup_device, load_and_prepare_model, interactive_loop, batch_process_file

parser = argparse.ArgumentParser(description='BERT inference test')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--interactive', action='store_true')
parser.add_argument('--masking_ratio', type=float, default=0.25)
parser.add_argument('--tokenizer', type=str, default="sapienzanlp/Minerva-350M-base-v1.0")
args = parser.parse_args()

device = setup_device()
model, config = load_and_prepare_model(
    args.model, 
    BERTModel, 
    device, 
    tokenizer=args.tokenizer
)

def show_output(input_text):
    """Display BERT inference results."""
    acc, out_tokens = model.inference_testing(input_text, masking_ratio=args.masking_ratio)
    print(f"\nInput: {input_text}")
    print(f"\nOutput: {out_tokens}")
    print(f"\nAccuracy: {acc:.2f}")

if args.interactive:
    interactive_loop(show_output)
elif args.prompt:
    show_output(args.prompt)
elif args.input and args.output:
    batch_process_file(
        args.input, 
        args.output, 
        model, 
        'inference_testing',
        masking_ratio=args.masking_ratio
    )
else:
    print("Provide --interactive OR --prompt OR both --input and --output")
