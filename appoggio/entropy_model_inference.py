import argparse
import torch
import pytorch_lightning as pl
from matformer.tokenizers import MatformerTokenizer
from matformer.model_config import ModelConfig  
from matformer.models import PL_ModelWrapper


parser = argparse.ArgumentParser(description='Inference on a Matformer model')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--length',type=int, default=100)
parser.add_argument('--temp', type=float, default=0.6)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--interactive', action=store_true)
parser.add_argument('--tokenizer', default=None)

args = parser.parse_args()
checkpoint_path = args.model 
model,config = PL_ModelWrapper.load_from_checkpoint(checkpoint_path, inference_fix=True)
print(f"Model loaded. Config = {config}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
    

generated_ids = model.generate(
        prompt=args.prompt, 
        max_length=args.length,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k
)

# For entropy model, Interactive model also supports Entropy computation and cut of text in patches
# By choosing the smoothing parameter (ex. > [CUT 0.3] "Text here" Will cut the text with a smoothing of 0.3

# If the model is a autoregressive, generate tokens
# If it's a masked, show the masked sequence and the predicted token for each masked position (wrong, true) plus a calculation of accuracy.
 
tokenizer = 'bytes' if args.tokenizer is None else AutoTokenizer.from_pretrained(args.tokenizer)
generation = tokenizer.decode(generated_ids.cpu().numpy())
print(f"Output: {generation}")

