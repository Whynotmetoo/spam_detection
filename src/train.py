import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Tuple, List

from dataset import load_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BertTrainer:
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        device: str = None
    ):
        """
        Initialize BERT trainer.
        
        Args:
            model_name (str): BERT model name
            max_length (int): Maximum sequence length
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate
            num_epochs (int): Number of training epochs
            device (str): Device to use for training
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Binary classification
        )
        self.model.to(self.device)
        
    def compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            preds (np.ndarray): Model predictions
            labels (np.ndarray): True labels
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average='binary'
        )
        accuracy = accuracy_score(labels, preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            optimizer (Optimizer): Optimizer
            scheduler (LRScheduler): Learning rate scheduler
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Tuple[float, Dict[str, float]]: Average validation loss and metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        metrics = self.compute_metrics(
            np.array(all_preds),
            np.array(all_labels)
        )
        
        return total_loss / len(val_loader), metrics
    
    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        output_dir: str = 'models/best_model'
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset): Validation dataset
            output_dir (str): Directory to save best model
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )
        
        logger.info(f"Created data loaders with batch size {self.batch_size}")
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=1e-8
        )
        
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        logger.info(f"Initialized optimizer with learning rate {self.learning_rate}")
        logger.info(f"Total training steps: {total_steps}")
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }
        
        best_f1 = 0
        
        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            
            # Evaluate
            val_loss, metrics = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(metrics['f1'])
            
            logger.info(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val F1: {metrics['f1']:.4f}, "
                f"Val Accuracy: {metrics['accuracy']:.4f}, "
                f"Val Precision: {metrics['precision']:.4f}, "
                f"Val Recall: {metrics['recall']:.4f}"
            )
            
            # Save best model
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                os.makedirs(output_dir, exist_ok=True)
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                logger.info(f"Saved new best model with F1 score: {best_f1:.4f}")
        
        return history

def main():
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets()
    
    # Initialize trainer
    trainer = BertTrainer(
        model_name='bert-base-uncased',
        max_length=512,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=3
    )
    
    # Train model
    history = trainer.train(train_dataset, val_dataset)
    
    logger.info("Training completed!")
    logger.info(f"Best F1 score: {max(history['val_f1']):.4f}")

if __name__ == '__main__':
    main() 