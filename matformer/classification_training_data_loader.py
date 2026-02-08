import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
from datasets import Dataset, Features, Value, ClassLabel, Sequence

class ClassificationTrainingDataLoader:
    """Load and validate data for model training with custom column mapping."""
    
    def __init__(
        self,
        filepath: Union[str, Path],
        text_column: str,
        label_column: Optional[str] = None,
        id_column: Optional[str] = None,
        additional_columns: Optional[List[str]] = None,
    ):
        """
        Args:
            filepath: Path to file
            text_column: Name of column containing text data
            label_column: Name of column containing labels (optional for inference)
            id_column: Name of column containing IDs (optional)
            additional_columns: List of other columns to load (optional)
        """
        self.filepath = Path(filepath)
        
        self.text_column = text_column
        self.label_column = label_column
        self.id_column = id_column
        self.additional_columns = additional_columns or []
        
        self._validate_and_load()
    
    def _validate_and_load(self):
        """Load CSV,TSV,JSON,CONLLU,CONLLX and validate required columns exist."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        # Load file
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
        required = [self.text_column]
        if self.label_column:
            required.append(self.label_column)
        if self.id_column:
            required.append(self.id_column)
        required.extend(self.additional_columns)
        
        missing = set(required) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available: {list(self.df.columns)}")
        
        # Remove rows with NaN in critical columns
        critical = [self.text_column]
        if self.label_column:
            critical.append(self.label_column)
        
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=critical)
        dropped = initial_len - len(self.df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing values in critical columns")
    
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
    
    def get_data(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns:
            texts: Array of text strings
            labels: Array of labels (None if label_column not specified)
            ids: Array of IDs (None if id_column not specified)
        """
        texts = self.df[self.text_column].values
        labels = self.df[self.label_column].values if self.label_column else None
        ids = self.df[self.id_column].values if self.id_column else None
        
        return texts, labels, ids
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __repr__(self) -> str:
        return f"TrainingDataLoader(file={self.filepath.name}, samples={len(self)}, columns={list(self.df.columns)})"
      
    def get_label_info(self) -> Tuple[np.ndarray, int]:
        """
        Returns:
            unique_labels: Sorted array of unique label values
            num_labels: Count of unique labels
        """
        if self.label_column is None:
            raise ValueError("Cannot get label info: label_column not specified")
        
        unique_labels = np.sort(self.df[self.label_column].unique())
        num_labels = len(unique_labels)
        
        return unique_labels, num_labels

    def get_num_labels(self) -> int:
        """Convenience method to get only the count of unique labels."""
        _, num_labels = self.get_label_info()
        return num_labels
      
    def to_hf_dataset(self) -> "Dataset":
        """
        Convert loaded data to HuggingFace Dataset format.
        
        Returns:
            Dataset object compatible with HuggingFace transformers
        """    
        
        if self.label_column is None:
            raise ValueError("Cannot create HF dataset: label_column not specified")
        
        # Determine task type and get configuration
        first_text = self.df[self.text_column].iloc[0]
        is_token_level = isinstance(first_text, list)
        
        if is_token_level:
            unique_labels, data_dict, features = self._prepare_token_classification()
        else:
            unique_labels, data_dict, features = self._prepare_sequence_classification()
        
        # Add ID column if present
        if self.id_column:
            data_dict["id"] = self.df[self.id_column].tolist()
            features["id"] = Value("string")
        
        # Create dataset with label mappings
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        dataset = Dataset.from_dict(data_dict, features=features)
        dataset.label2id = label2id
        dataset.id2label = {idx: label for label, idx in label2id.items()}
        
        return dataset

    def _prepare_token_classification(self):
        """Helper for token-level tasks (POS, NER)."""
        # Get all unique labels across all sentences
        all_labels = [label for labels in self.df[self.label_column] for label in labels]
        unique_labels = sorted(set(all_labels))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        
        data_dict = {
            "tokens": self.df[self.text_column].tolist(),
            "tags": [
                [label2id[label] for label in labels]
                for labels in self.df[self.label_column]
            ]
        }
        
        features = Features({
            "tokens": Sequence(Value("string")),
            "tags": Sequence(ClassLabel(names=unique_labels))
        })
        
        return unique_labels, data_dict, features

    def _prepare_sequence_classification(self):
        """Helper for sequence-level tasks (sentiment, topic)."""
        unique_labels = sorted(self.df[self.label_column].unique())
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        
        data_dict = {
            "text": self.df[self.text_column].tolist(),
            "label": [label2id[label] for label in self.df[self.label_column]]
        }
        
        features = Features({
            "text": Value("string"),
            "label": ClassLabel(names=unique_labels)
        })
        
        return unique_labels, data_dict, features