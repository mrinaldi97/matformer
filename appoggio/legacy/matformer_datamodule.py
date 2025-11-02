"""
THIS FILE IS DEPRECATED AND IS GOING TO BE REMOVED
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from matformer.model_config import ModelConfig  
from torch_atlas_ds import AtlasDataset, AtlasDatasetWriter  

#from matformer.matformer_dataset import MatformerDataset

class VECCHIOMatformerDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size, tokenizer, config, num_workers=0, autoencoder_experiment=False):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config
        self.tokenizer = tokenizer

        self.type = 'mdat' if os.path.exists(os.path.join(data_root, 'matformer_dataset.mdat')) else 'atlas'
        if self.type == 'mdat':
            print("È UN MDAT!")

        print(f"Max pos emb. in data module: {self.config.max_position_embeddings}")

        if self.tokenizer.tokenizer_modality == 'bytes':
            self.modality = 'bytes'
            print("DEBUG: DATA MODULE IN BYTES MODALITY")
        else:
            self.modality = 'tokens'

        # Ha l'entropia?
        self.entropy_function = None
        self.entropy_model = None
        self.entropy_smoothing = None
        self.n_bytes=self.config.max_position_embeddings
        self.chunk_size=None
        if getattr(self.config, "has_text_autencoder", True) or autoencoder_experiment:
            if self.config.has_text_autoencoder is not None or autoencoder_experiment:
                self.modality = 'entropy_patches'
                print("Stiamo usando la modalità sperimentale autoencoder.")
                import sys
                sys.path.append('../')
                from inference import load_inference_model, compute_entropy
                self.entropy_model,entropy_cfg = load_inference_model(self.config.entropy.entropy_model_path, ModelClass=EntropyModel, tokenizer='bytes', map_location=torch.device('cuda'))
                self.entropy_smoothing = self.config.entropy.entropy_smoothing
                self.entropy_function = compute_entropy
                self.n_bytes=self.config.encoder.max_position_embeddings
                self.chunk_size=self.config.max_position_embeddings
        # Dataset
        self.dataset = MatformerDataset(
            path=self.data_root,
            modality=self.modality,
            tokens=self.config.max_position_embeddings,
            n_bytes=self.n_bytes,
            byte_tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            entropy_function=self.entropy_function,
            entropy_model=self.entropy_model,
            entropy_smoothing=self.entropy_smoothing
        )
    def setup(self, stage=None):
        if self.type=='atlas':
            self.dataset = AtlasDataset(self.data_root)
        else:
            self.dataset = MatformerDataset(
                path=self.data_root,
                modality=self.modality,
                tokens=self.config.max_position_embeddings,
                n_bytes=self.n_bytes,
                byte_tokenizer=self.tokenizer,
                chunk_size=self.chunk_size,
                entropy_function=self.entropy_function,
                entropy_model=self.entropy_model,
                entropy_smoothing=self.entropy_smoothing
            )
    

    def _collate_fn_old(self, batch):
        texts = [item['text'] for item in batch]
        tokenized=dict()
        tokenized['input_ids'] = self.tokenizer.batch_encode(texts)
        tokenized['text'] = texts #I add back the raw text, this is currently required in the BLT model
        return tokenized
        
    def collate_fn(self, batch):
        if self.modality == 'entropy_patches':
            #  batch is a list (len: batch_size) containing list of strings (patches)
            
            sequence = self.tokenizer.batch_encode(batch)
            # Sequence after tokenizer is torch.Size([B, seq_len(P), encoder_seq_len(S)])
            
        else:
            # Here batch is list of token sequences
            sequence = self.tokenizer.process_pretokenized_batch(
                token_sequences=batch,
                config=self.config,
                varlen_strategy=self.tokenizer.varlen_strategy,
                pad_token_id=self.config.pad_token_id,
            )
        return {'input_ids': sequence}     
    def __len__(self):
        #Molto temporaneamente, apro e richiudo l'atlas dataset solo per calcolarne la lunghezza; questo perchè l'informazione serve prima del setup.
        if self.type=='atlas':
            return len(AtlasDataset(self.data_root))
        else:
            return self.dataset.__len__()
            
    """ 
    def state_dict(self):
        state = {"current_train_batch_index": self.current_train_batch_index}
        return state

    def load_state_dict(self, state_dict):
        self.current_train_batch_index = state_dict["current_train_batch_index"]
    """
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self.collate_fn,  
            shuffle=False  
        )
