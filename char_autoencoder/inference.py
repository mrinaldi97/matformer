import sys, argparse, random, datetime, csv, json
sys.path.append('../')
import torch
from collections import defaultdict
from matformer.tokenizers import ByteTokenizer
from matformer.model_config import ModelConfig
from autoencoders import TransCharAutoencoder
from matformer.tensors_dataclasses import PaddedTensor, NormalTensor, UnpaddedTensor

def decode_tokens(tokens):
    """Decode tokens back to text using the same method as training."""
    try:
        # Convert tokens to bytes, handling any invalid values
        byte_values = []
        for t in tokens:
            if 0 <= t <= 256:
                byte_values.append(t)
            # Skip invalid tokens
        return bytes(byte_values).decode('utf-8', errors='replace')
    except:
        return ''.join(chr(t) if 0 <= t <= 260 else '?' for t in tokens)

# inference_reconstructor.py
def load_model(checkpoint_path, config_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config from JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    encoder_config = ModelConfig(**config['encoder'])
    decoder_config = ModelConfig(**config['decoder'])
    
    # Create model with same pattern as training script
    # Don't pass device to constructor, handle it after loading
    model = TransCharAutoencoder(encoder_config, decoder_config, device='cpu')
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    
    # Clean state dict - remove "model." prefix but keep .module. for ModuleWrapper components
    cleaned_sd = {}
    print("Processing checkpoint keys...")
    for key, value in sd.items():
        # Remove "model." prefix if present
        clean_key = key
        if clean_key.startswith('model.'):
            clean_key = clean_key[6:]  # Remove "model."
        
        # Don't remove .module. - the inference model expects it due to ModuleWrapper
        print(f"  {key} -> {clean_key}")
        cleaned_sd[clean_key] = value
    
    print(f"\nLoading {len(cleaned_sd)} parameters...")
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_sd, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    # Move model to correct device after loading weights
    model = model.to(device)
    
    if device == 'cuda':
        if encoder_config.attn_impl == "flash" and torch.cuda.is_bf16_supported():
            model = model.bfloat16()
        else:
            model = model.half()
    
    # Fix ALiBi slopes to fp32 after precision conversion
    for module in model.modules():
        if hasattr(module, "alibi_slopes") and module.alibi_slopes is not None:
            module.alibi_slopes = module.alibi_slopes.to(dtype=torch.float32)
    
    model.eval()
    print(f"Model loaded from {checkpoint_path} on {device}")
    return model, device, encoder_config

def encode_text(text_or_bytes, max_len, pad_token, device):
    """Encode text using the same method as training (UTF-8 bytes)."""
    if isinstance(text_or_bytes, str):
        # Use same encoding as training: UTF-8 bytes clamped to 0-255
        l = [x for x in text_or_bytes.encode('utf-8')]
        l.append(258)
        encoded = torch.tensor(l, dtype=torch.int32)
    elif isinstance(text_or_bytes, list) and all(isinstance(x, int) for x in text_or_bytes):
        encoded = torch.tensor(text_or_bytes, dtype=torch.int32).clamp(0, 260)
    else:
        raise TypeError("Input must be str or list[int].")
    
    length = min(len(encoded), max_len)
    encoded = encoded[:length]
    
    # Pad if necessary
    if length < max_len:
        encoded = torch.cat([encoded, torch.full((max_len - length,), 258, dtype=torch.int32)])
    
    inp = encoded.unsqueeze(0).to(device, dtype=torch.long)
    length_tensor = torch.tensor([[length]], device=device, dtype=torch.long)
    
    return NormalTensor(tensor=inp), NormalTensor(tensor=length_tensor)


def reconstruct_from_text(model, input_text, device, max_len, pad_token):
    """Reconstruct text using EXACT training conditions for debugging."""
    
    # CRITICAL: Set model to training mode to match training conditions
    original_training_state = model.training
    model.train()
    
    print("=== TRAINING MODE RECONSTRUCTION DEBUG ===")
    
    with torch.no_grad():  # No gradients but training mode
        inp, lengths = encode_text(input_text, max_len, pad_token, device)
        orig_len = lengths.tensor.item()
        
        print(f"Input length: {orig_len}")
        input_tokens = inp.tensor.squeeze().tolist()[:orig_len]
        print(f"Input tokens: {input_tokens}")
        print(f"Input chars: {[chr(t) if 0 <= t <= 255 else f'<{t}>' for t in input_tokens]}")
        
        # Use EXACT same forward pass as training_step
        char_logits, seqlen_logits = model(inp, lengths)
        
        # Compute same targets as training for comparison
        seqlen_targets = (lengths.tensor.squeeze(-1) - 1).long()
        char_targets = inp.tensor.view(-1).long()
        
        #print(f"Sequence length target (training): {seqlen_targets.item()}")
        #print(f"Sequence length prediction: {seqlen_logits.tensor.argmax(-1).item()}")
        
        # Get character predictions
        char_predictions = char_logits.tensor.argmax(-1).squeeze(0)
        predicted_tokens = char_predictions.tolist()[:orig_len]
        
        print(f"Predicted tokens: {predicted_tokens}")
        print(f"Predicted chars: {[chr(t) if 0 <= t <= 255 else f'<{t}>' for t in predicted_tokens]}")
        
        # Check token-by-token accuracy (same as training)
        matches = [pred == actual for pred, actual in zip(predicted_tokens, input_tokens)]
        accuracy = sum(matches) / len(matches) if matches else 0
        
        print(f"Token-by-token matches: {matches}")
        print(f"Training-mode accuracy: {accuracy:.4f} ({sum(matches)}/{len(matches)})")
        
        # Show differences
        if accuracy < 1.0:
            print("MISMATCHES:")
            for i, (pred, actual, match) in enumerate(zip(predicted_tokens, input_tokens, matches)):
                if not match:
                    pred_char = chr(pred) if 0 <= pred <= 255 else f'<{pred}>'
                    actual_char = chr(actual) if 0 <= actual <= 255 else f'<{actual}>'
                    print(f"  Position {i}: predicted '{pred_char}' ({pred}) != actual '{actual_char}' ({actual})")
        
        # Decode the output
        rec_text = decode_tokens(predicted_tokens)
        
        print(f"Reconstructed text: '{rec_text}'")
        print(f"Original text: '{input_text}'")
        print(f"Perfect reconstruction: {rec_text == input_text}")
        
    # Restore original model state
    model.train(original_training_state)
    
    return rec_text, orig_len, orig_len


def run_random_tests(model, device, num_samples_per_len, max_len, pad_token):
    print(f"\n--- Random Tests: {num_samples_per_len} per length 1–{max_len} ---")
    stats = defaultdict(lambda: [0,0,0])
    details = []
    for L in range(1, max_len+1):
        for _ in range(num_samples_per_len):
            orig = [random.randrange(256) for _ in range(L)]
            txt = bytes(orig).decode('latin-1')
            rec, oL, pL = reconstruct_from_text(model, txt, device, max_len, pad_token)
            try:
                rec_b = list(rec.encode('latin-1'))
            except:
                rec_b = []
            ok_r = (orig == rec_b)
            ok_l = (oL == pL)
            stats[L][0] += 1; stats[L][1] += ok_r; stats[L][2] += ok_l
            details.append({
                "length": L,
                "original": orig,
                "reconstructed": rec_b,
                "orig_len": oL,
                "pred_len": pL,
                "ok_recon": ok_r,
                "ok_len": ok_l
            })

    print("\nLen | Recon% | Len% | N")
    for L in sorted(stats):
        tot, cr, cl = stats[L]
        print(f"{L:>3} | {cr/tot*100:6.2f}% | {cl/tot*100:6.2f}% | {tot}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sumf = f"reconstruction_summary_{ts}.csv"; detf = f"reconstruction_details_{ts}.csv"
    with open(sumf,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(["length","recon%","len%","total"])
        for L in sorted(stats):
            tot,cr,cl = stats[L]
            w.writerow([L,f"{cr/tot*100:.2f}",f"{cl/tot*100:.2f}",tot])
    with open(detf,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        w.writeheader(); w.writerows(details)
    print(f"Saved summary to {sumf} and details to {detf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--config", required=True, help="Config JSON file path")
    parser.add_argument("--random_tests_samples", type=int, default=0)
    args = parser.parse_args()

    model, device, encoder_config = load_model(args.model, args.config)
    
    max_len = encoder_config.max_position_embeddings
    pad_token = encoder_config.pad_token_id
    
    print(f"Model info:")
    print(f"  Max length: {max_len}")
    print(f"  PAD token: {pad_token}")
    print(f"  Vocab size: {encoder_config.vocab_size}")

    print(f"Ready (MAX_LEN={max_len}). Commands: r <text>, t <N>, exit.")
    if args.random_tests_samples > 0:
        run_random_tests(model, device, args.random_tests_samples, max_len, pad_token)

    while True:
        cmd = input(">> ").strip()
        if cmd in ("exit", "quit"):
            break
        if cmd.startswith("r "):
            txt = cmd[2:]
            rec, oL, pL = reconstruct_from_text(model, txt, device, max_len, pad_token)
            print(f"Orig(len={oL}): \"{txt}\"\nRec (len={pL}): \"{rec}\"")
        elif cmd.startswith("t "):
            try:
                run_random_tests(model, device, int(cmd[2:]), max_len, pad_token)
            except ValueError:
                print("t expects an integer.")
        else:
            print("Unknown—use r <text>, t <N>, or exit.")

if __name__=="__main__":
    main()
