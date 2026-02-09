import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
from datasets import load_dataset, Dataset

class ClassificationTrainingDataLoader:
    """Load and validate data for model training with custom column mapping."""
    
    def __init__(
        self,
        text_column: str,
        label_column: str,
        filepath: Optional[Union[str, Path]] = None,
        id_column: Optional[str] = None,
        additional_columns: Optional[List[str]] = None,
        hf_dataset: Optional[str] = None,
        hf_split: str = "train",            
        hf_config: Optional[str] = None,    
    ):
        """
        Args:
            filepath: Path to file
            text_column: Name of column containing text data
            label_column: Name of column containing labels (optional for inference)
            id_column: Name of column containing IDs (optional)
            additional_columns: List of other columns to load (optional)
            hf_dataset: HuggingFace dataset name (alternative to filepath)
            hf_split: HuggingFace split to load (default: "train")
            hf_config: HuggingFace dataset configuration (optional)
        """
        self.filepath = Path(filepath) if filepath else None
        
        self.text_column = text_column
        self.label_column = label_column
        self.id_column = id_column
        self.additional_columns = additional_columns or []
        
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split
        self.hf_config = hf_config
        
        # Validate source specification
        if not self.hf_dataset and not self.filepath:
            raise ValueError("Must specify either 'filepath' or 'hf_dataset'")
        if self.hf_dataset and self.filepath:
            raise ValueError("Cannot specify both 'filepath' and 'hf_dataset'")
        
        self._validate_and_load()
    
    def _validate_and_load(self):
        """Load CSV,TSV,JSON,CONLLU,CONLLX and validate required columns exist."""
        
        print("\n--- WARNING ---\nAs of now, the loader loads all data in memory\n\n")
        
        if self.hf_dataset:
            try:
                dataset = load_dataset(
                    self.hf_dataset,
                    name=self.hf_config,
                    split=self.hf_split
                )
                self.df = dataset.to_pandas()
            except Exception as e:
                raise ValueError(f"Failed to load HF dataset '{self.hf_dataset}': {e}") from e
        
        # Load from file
        else:
            try:
                suffix = self.filepath.suffix.lower()
                
                if suffix in ['.conllu', '.conll']:
                    self.df = self._load_conll(suffix == '.conllu')
                elif suffix == '.csv':
                    self.df = pd.read_csv(self.filepath)
                elif suffix == '.tsv':
                    self.df = pd.read_csv(self.filepath, sep='\t')
                elif suffix == '.json':
                    self.df = pd.read_json(self.filepath)
                else:
                    raise ValueError(f"Unsupported format: {suffix}")
            except (pd.errors.ParserError, FileNotFoundError) as e:
                raise ValueError(f"Failed to load {self.filepath}: {e}") from e
        
        # Validate columns
        required = [self.text_column, self.label_column]
        if self.id_column:
            required.append(self.id_column)
        required.extend(self.additional_columns)
        
        missing = set(required) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available: {list(self.df.columns)}")
        
        # Remove rows with NaN in required columns  
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=required)
        dropped = initial_len - len(self.df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing values in required columns")
    
    def _load_conll(self, is_conllu: bool):
        """Parse CoNLL-U or CoNLL-X to DataFrame with tokens and labels per sentence."""
        sentences = []
        tokens, labels = [], []
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and metadata
                if not line or line.startswith('#'):
                    if tokens:  # End of sentence
                        sentences.append({
                            self.text_column: tokens,
                            self.label_column: labels
                        })
                        tokens, labels = [], []
                    continue
                
                parts = line.split('\t')
                
                # Skip multiword tokens (CoNLL-U: IDs like "1-2")
                if '-' in parts[0]:
                    continue
                
                # Extract token (column 1) and POS tag (column 3 for UPOS in CoNLL-U, column 4 in CoNLL-X)
                token = parts[1]
                pos_tag = parts[3] if is_conllu else parts[4]
                
                tokens.append(token)
                labels.append(pos_tag)
        
        # Add final sentence if exists
        if tokens:
            sentences.append({
                self.text_column: tokens,
                self.label_column: labels
            })
        
        return pd.DataFrame(sentences)
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Returns:
            texts: Array of text strings
            labels: Array of labels
            ids: Array of IDs (None if id_column not specified)
        """
        texts = self.df[self.text_column].values
        labels = self.df[self.label_column].values if self.label_column else None
        ids = self.df[self.id_column].values if self.id_column else None
        
        return texts, labels, ids
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __repr__(self) -> str:
        source = f"hf={self.hf_dataset}" if self.hf_dataset else f"file={self.filepath.name}"
        return f"TrainingDataLoader({source}, samples={len(self)}, columns={list(self.df.columns)})"
 
    def get_label_info(self) -> Tuple[np.ndarray, int]:
        """
        Returns:
            unique_labels: Sorted array of unique label values
            num_labels: Count of unique labels
        """       
        labels = self.df[self.label_column]
            
        # Check if labels are lists (token classification) or scalars (sequence classification)
        if isinstance(labels.iloc[0], (list, np.ndarray)):
          all_labels = [label for label_list in labels for label in label_list]
          unique_labels = np.sort(np.unique(all_labels))
        else:
          unique_labels = np.sort(labels.unique())
            
        num_labels = len(unique_labels)
            
        return unique_labels, num_labels

    def get_num_labels(self) -> int:
        """Convenience method to get only the count of unique labels."""
        _, num_labels = self.get_label_info()
        return num_labels