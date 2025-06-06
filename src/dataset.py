import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
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
        subjects: List[str],
        bodies: List[str],
        labels: List[int],
        tokenizer: RobertaTokenizer,
        max_length: int = 512
    ):
        """
        Dataset class for email spam detection.
        
        Args:
            subjects (List[str]): List of email subjects
            bodies (List[str]): List of email bodies
            labels (List[int]): List of labels (0 for ham, 1 for spam)
            tokenizer (RobertaTokenizer): RoBERTa tokenizer
            max_length (int): Maximum sequence length
        """
        self.subjects = subjects
        self.bodies = bodies
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.subjects)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subject = self.clean_text(str(self.subjects[idx]))
        body = self.clean_text(str(self.bodies[idx]))
        label = self.labels[idx]
        
        # Handle empty subject/body
        if not subject.strip():
            subject = "[NO SUBJECT]"
        if not body.strip():
            body = "[NO BODY]"
        
        # Combine subject and body with clear separators
        text = f"<s> [SUBJECT] {subject} [BODY] {body} </s>"
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,  # We already added special tokens
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

def load_datasets(
    train_path: str = 'datasets/train.csv',
    val_path: str = 'datasets/validate.csv',
    test_path: str = 'datasets/test.csv',
    model_name: str = 'roberta-base',
    max_length: int = 512
) -> Tuple[EmailDataset, EmailDataset, EmailDataset]:
    """
    Load train, validation, and test datasets.
    
    Args:
        train_path (str): Path to training data CSV
        val_path (str): Path to validation data CSV
        test_path (str): Path to test data CSV
        model_name (str): RoBERTa model name
        max_length (int): Maximum sequence length
        
    Returns:
        Tuple[EmailDataset, EmailDataset, EmailDataset]: Train, validation, and test datasets
    """
    try:
        # Load data
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Loaded {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
        
        # Initialize tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Create datasets
        train_dataset = EmailDataset(
            subjects=train_df['subject'].values,
            bodies=train_df['body'].values,
            labels=train_df['label'].values,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        val_dataset = EmailDataset(
            subjects=val_df['subject'].values,
            bodies=val_df['body'].values,
            labels=val_df['label'].values,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        test_dataset = EmailDataset(
            subjects=test_df['subject'].values,
            bodies=test_df['body'].values,
            labels=test_df['label'].values,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise 