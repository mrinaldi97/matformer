import argparse
import torch
import pytorch_lightning as pl
from matformer.tokenizers import MatformerTokenizer
from matformer.model_config import ModelConfig  
from matformer.models import Autoregressive_Model


parser = argparse.ArgumentParser(description='Inference on the GPT model')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--length',type=int, default=100)
parser.add_argument('--temp', type=float, default=0.6)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--tokenizer',type=str,default="sapienzanlp/Minerva-350M-base-v1.0")

args = parser.parse_args()
checkpoint_path = args.model 
model,config = Autoregressive_Model.load_from_checkpoint(checkpoint_path, inference_fix=True, tokenizer=args.tokenizer)
print(f"Model loaded. Config = {config}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device, torch.bfloat16)
   

generated_ids = model.generate(
        prompt=args.prompt, 
        max_length=args.length,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k
)
    
tokenizer = model.tokenizer
generation = tokenizer.decode(generated_ids)
print(f"Output: {generation}")

