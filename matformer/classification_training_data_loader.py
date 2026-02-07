import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np

class ClassificationTrainingDataLoader:
    """Load and validate CSV data for model training with custom column mapping."""
    
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
            filepath: Path to CSV file
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
        """Load CSV and validate required columns exist."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        # Load file
        try:
            suffix = self.filepath.suffix.lower()
            
            if suffix == '.csv':
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
    
    def get_dataframe(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Return DataFrame with specified columns or all loaded columns."""
        if columns:
            return self.df[columns].copy()
        
        cols = [self.text_column]
        if self.label_column:
            cols.append(self.label_column)
        if self.id_column:
            cols.append(self.id_column)
        cols.extend(self.additional_columns)
        
        return self.df[cols].copy()
    
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
      
    def get_label_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            unique_labels: Sorted array of unique label values
            counts: Frequency of each label
        """
        if self.label_column is None:
            raise ValueError("Cannot get label distribution: label_column not specified")
        
        unique_labels, counts = np.unique(self.df[self.label_column].values, return_counts=True)
        return unique_labels, counts