import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 512
    ):
        """
        Dataset class for email spam detection.
        
        Args:
            texts (List[str]): List of email texts
            labels (List[int]): List of labels (0 for ham, 1 for spam)
            tokenizer (BertTokenizer): BERT tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_split_data(
    data_path: str,
    train_ratio: float = 0.8,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split data into train and validation sets.
    
    Args:
        data_path (str): Path to processed CSV file
        train_ratio (float): Ratio of training data
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and validation DataFrames
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")
        
        # Shuffle and split
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        train_size = int(len(df) * train_ratio)
        
        train_df = df[:train_size]
        val_df = df[train_size:]
        
        logger.info(f"Split data into {len(train_df)} training and {len(val_df)} validation samples")
        
        return train_df, val_df
        
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {str(e)}")
        raise

def create_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = 'bert-base-uncased',
    max_length: int = 512
) -> Tuple[EmailDataset, EmailDataset]:
    """
    Create train and validation datasets.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        model_name (str): BERT model name
        max_length (int): Maximum sequence length
        
    Returns:
        Tuple[EmailDataset, EmailDataset]: Train and validation datasets
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    train_dataset = EmailDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = EmailDataset(
        texts=val_df['text'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return train_dataset, val_dataset 