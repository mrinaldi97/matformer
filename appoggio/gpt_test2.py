import argparse
from matformer.models import Autoregressive_Model
from inference.inference_code import setup_device, load_and_prepare_model, interactive_loop, batch_process_file

parser = argparse.ArgumentParser(description='Test di un modello GPT-Like')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--interactive', action='store_true')
parser.add_argument('--max_length', type=int, default=100)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--tokenizer', type=str, default="sapienzanlp/Minerva-350M-base-v1.0")
args = parser.parse_args()

device = setup_device()
model, config = load_and_prepare_model(
    args.model, 
    Autoregressive_Model, 
    device, 
    tokenizer=args.tokenizer
)

def show_output(input_text):
    generated_text = model.generate(
        input_text,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    print(f"\nInput: {input_text}")
    print(f"\nGenerated: {generated_text}")

def batch_process_gpt(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    with open(output_file, "w") as f:
        for line in lines:
            line = line.strip()
            generated = model.generate(
                line,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            f.write(f"Input: {line}\nGenerated: {generated}\n\n")
    
    print(f"Saved to {output_file}")

if args.interactive:
    interactive_loop(show_output)
elif args.prompt:
    show_output(args.prompt)
elif args.input and args.output:
    batch_process_gpt(args.input, args.output)
else:
    print("Provide --interactive OR --prompt OR both --input and --output")
