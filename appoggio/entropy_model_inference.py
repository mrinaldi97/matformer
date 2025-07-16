import argparse
import torch
import pytorch_lightning as pl
from matformer.tokenizers import ByteLevelTokenizer
from matformer.model_config import ModelConfig  
from matformer.models import EntropyModel


parser = argparse.ArgumentParser(description='Inference on the Byte-level entropy model')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--length',type=int, default=100)
parser.add_argument('--temp', type=float, default=0.6)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--top_p', type=float, default=0.9)
args = parser.parse_args()
checkpoint_path = args.model 
model,config = EntropyModel.load_from_checkpoint(checkpoint_path, inference_fix=True)
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
    
tokenizer = ByteLevelTokenizer(config)
generation = tokenizer.decode(generated_ids.cpu().numpy())
print(f"Output: {generation}")

