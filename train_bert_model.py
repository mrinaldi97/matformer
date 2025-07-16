"""
Implementation of the Entropy Model
"""
import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from matformer.tokenizers import MatformerTokenizer
from matformer.metrics import BitsPerByte  
from matformer.training_functions import MatformerDataModule
from matformer.model_config import ModelConfig  
from matformer.models import EntropyModel
import signal
import sys
from transformers import AutoTokenizer
def main():
    config_big_model_1024_window = ModelConfig(
        name='BERT Model',
        hidden_dim=768,
        ffn_factor=1.0,
        n_layers=14,
        n_heads=12,
        vocab_size=32768,
        bos_id = 1,
        eos_id = 2,
        pad_id = 0,      
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        attention_type=['causal','sliding'],
        sliding_window_size=512,
        sliding_layers=[0,1,2,3,4,6,7,9,10,11],
        sliding_type='partial',
        max_seqlen=1024,
        block_size_for_attention=128,
        compile_flexattn=False,
        bias=False,
        training_objective='masked'
    ) 
    config=config_big_model_1024_window
    parser = argparse.ArgumentParser(description='Train byte-level entropy model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--varlen_strategy', type=str, required=True, choices=['padding','unpadding','nested'])    
    parser.add_argument('--attn_impl', type=str, required=True, choices=['sdpa','flash','flex'])        
    parser.add_argument('--dump_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()
    config['attn_impl']=args.attn_impl
    train_config = {
        "lr": 3e-4,
        "max_steps": args.max_steps
    }
    """train_config = {
        "lr": 1e-5,
        "warmup_steps": 300,
        "max_steps": args.max_steps
    }  """  
    pl.seed_everything(27)
    tok=AutoTokenizer.from_pretrained("sapienzanlp/Minerva-350M-base-v1.0")

    tokenizer = MatformerTokenizer(config,tokenizer=tok,varlen_strategy=args.varlen_strategy)
    data_module = MatformerDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=2,
        config=config,
        tokenizer=tokenizer
    )

    model = EntropyModel(config=config, train_config=train_config, device='cuda')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.dump_dir, 
        filename='liberliber_1024',  
        save_top_k=1,  
        save_last=True  
    )
    trainer = pl.Trainer(
        default_root_dir=args.dump_dir,
        max_steps=args.max_steps,
        precision='16-mixed',
        log_every_n_steps=100,
        accumulate_grad_batches=1,
        accelerator='gpu',
        max_epochs=1,
        callbacks=[checkpoint_callback]  
    )
    torch.set_float32_matmul_precision("high")
    #model = torch.compile(model)
    #print("Model compiled.")
#    def save_ctrlc(sig, frame):
#        print('\nCTRL+C Pressed, saving intermediate checkpoint')
#        trainer.should_stop = True  
#    signal.signal(signal.SIGINT, save_ctrlc)
    trainer.fit(model, data_module)
    
if __name__ == '__main__':
    main()
