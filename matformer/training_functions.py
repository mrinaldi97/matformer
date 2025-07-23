import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_atlas_ds import AtlasDataset
from matformer.model_config import ModelConfig  

class MatformerDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size, tokenizer, config, num_workers=0):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config=config
        self.tokenizer=tokenizer

    def setup(self, stage=None):
        self.dataset = AtlasDataset(self.data_root)
    def _collate_fn(self, batch):
        texts = [item['text'] for item in batch]
        tokenized=dict()
        tokenized['input_ids'] = self.tokenizer.batch_encode(texts)
        tokenized['text'] = texts #I add back the raw text, this is currently required in the BLT model
        return tokenized

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn 
        )
