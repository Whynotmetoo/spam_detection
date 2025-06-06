import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Tuple, List
from dataset import EmailDataset
import pandas as pd

from dataset import load_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobertaTrainer:
    def __init__(
        self,
        model_name: str = 'roberta-base',
        max_length: int = 512,
        batch_size: int = 16,
        learning_rate: float = 2e-5,  # Reduced learning rate
        num_epochs: int = 10,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        device: str = None,
        gradient_accumulation_steps: int = 4
    ):
        """
        Initialize RoBERTa trainer.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS (Metal Performance Shaders) device")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA device")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            self.device = torch.device(device)
            logger.info(f"Using specified device: {device}")
        
        # Initialize model and tokenizer
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs('model', exist_ok=True)
    
    def compute_metrics(self, labels, preds):
        """
        Compute evaluation metrics.
        """
        # Ensure inputs are numpy arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Handle special case of all-zero predictions
        if np.all(preds == 0):
            precision = recall = f1 = 0.0
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='binary', zero_division=0
            )
        
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
    
    def train(self, train_dataset, val_dataset=None):
        """
        Train the model.
        """
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size
            )
        
        # Log dataset sizes
        logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Initialize loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        total_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Check model parameters (only at the start of first epoch)
        if True:  # Changed to True to check parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.info(f"Parameter: {name} | Shape: {param.shape} | Mean: {param.data.mean():.6f} | Std: {param.data.std():.6f}")
        
        best_f1 = 0
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Clear gradients at the start of gradient accumulation
                if batch_idx % self.gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                
                # Debug output (only for first few batches of first epoch)
                if epoch == 0 and batch_idx < 3:
                    probs = torch.softmax(logits, dim=1)
                    logger.info(f"Batch {batch_idx} sample:")
                    logger.info(f"Logits: {logits[0].detach().cpu().tolist()}")
                    logger.info(f"Probabilities: {probs[0].detach().cpu().tolist()}")
                    logger.info(f"True label: {labels[0].item()}")
                
                # Compute loss
                loss = criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item() * self.gradient_accumulation_steps
                
                # Update parameters after gradient accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # Check gradients
                    for name, param in self.model.named_parameters():
                        if 'classifier' in name and param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            if grad_norm < 1e-6:
                                logger.warning(f"Low gradient: {name} = {grad_norm}")
                
                progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
            
            avg_train_loss = total_loss / len(train_dataloader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_dataset:
                self.model.eval()
                all_val_preds = []
                all_val_labels = []
                val_loss = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc="Validation"):
                        # Move batch to device
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        # Forward pass
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        logits = outputs.logits
                        
                        # Compute loss
                        loss = criterion(logits, labels)
                        val_loss += loss.item()
                        
                        # Get predictions
                        preds = torch.argmax(logits, dim=1)
                        all_val_preds.append(preds)
                        all_val_labels.append(labels)
                
                # Combine all batch results
                val_preds = torch.cat(all_val_preds)
                val_labels = torch.cat(all_val_labels)
                
                # Compute metrics
                metrics = self.compute_metrics(val_labels, val_preds)
                avg_val_loss = val_loss / len(val_dataloader)
                logger.info(f"Validation loss: {avg_val_loss:.4f}")
                logger.info(f"Validation metrics: {metrics}")
                
                # Save best model
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    model_path = 'model/best_model'
                    self.model.save_pretrained(model_path)
                    self.tokenizer.save_pretrained(model_path)
                    logger.info(f"Saved best model (F1={best_f1:.4f}) to {model_path}")
                
                # Always save the latest model
                model_path = 'model/latest_model'
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                logger.info(f"Saved latest model to {model_path}")

def main():
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Load datasets
    train_df = pd.read_csv('datasets/train.csv')
    val_df = pd.read_csv('datasets/validate.csv')
    
    train_dataset = EmailDataset(
        subjects=train_df['subject'].values,
        bodies=train_df['body'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=512
    )
    
    val_dataset = EmailDataset(
        subjects=val_df['subject'].values,
        bodies=val_df['body'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Initialize trainer
    trainer = RobertaTrainer()
    
    # Train model
    trainer.train(train_dataset, val_dataset)

if __name__ == "__main__":
    main() 