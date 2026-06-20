import torch
from matformer.tokenizers import MatformerTokenizer
from matformer.model_config import ModelConfig

def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_model(model_path, model_class, device, tokenizer=None, **kwargs):
    if tokenizer:
        model, config = model_class.load_from_checkpoint(
            model_path, 
            inference_fix=True, 
            map_location=device, 
            tokenizer=tokenizer,
            **kwargs
        )
    else:
        model, config = model_class.load_from_checkpoint(
            model_path, 
            inference_fix=True, 
            map_location=device,
            **kwargs
        )
    
    model = model.to(device, torch.bfloat16)
    model.eval()
    return model, config

def interactive_loop(inference_function):
    while True:
        try:
            inp = input(">>> ")
            if not inp:
                continue
            inference_function(inp)
        except KeyboardInterrupt:
            print("\nFine.")
            break

def batch_process_file(input_file, output_file, model, inference_method, **kwargs):
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    with open(output_file, "w") as f:
        for line in lines:
            line = line.strip()
            if hasattr(model, inference_method):
                if inference_method == 'inference_testing':
                    acc, predicted = getattr(model, inference_method)(line, **kwargs)
                    out = " ".join(f"[{tok}]" for tok in predicted)
                    f.write(f"{line}\n{out}\n\n")
                else:
                    result = getattr(model, inference_method)(line, **kwargs)
                    f.write(f"{line}\n{result}\n\n")
    
