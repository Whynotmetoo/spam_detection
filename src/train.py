import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
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
import json
import csv

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
        num_epochs: int = 10,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        device: str = None,
        gradient_accumulation_steps: int = 4,
        patience: int = 3  # Early stopping patience
    ):
        """
        Initialize BERT trainer.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.patience = patience
        
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
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Register special tokens
        special_tokens_dict = {'additional_special_tokens': ['[SUBJECT]', '[BODY]']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        if num_added_toks > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
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
        
        history = []  # For saving metrics per epoch
        best_f1 = 0
        best_epoch = 0
        epochs_no_improve = 0
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
                
                # Save history
                history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1']
                })
                
                # Early stopping logic
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    model_path = 'model/best_model'
                    self.model.save_pretrained(model_path)
                    self.tokenizer.save_pretrained(model_path)
                    logger.info(f"Saved best model (F1={best_f1:.4f}) to {model_path}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"No improvement in F1 for {epochs_no_improve} epoch(s)")
                
                # Always save the latest model
                model_path = 'model/latest_model'
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                logger.info(f"Saved latest model to {model_path}")
                
                # Check early stopping
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (no F1 improvement for {self.patience} epochs)")
                    break
        
        # Save training parameters
        params = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'patience': self.patience,
            'best_f1': best_f1,
            'best_epoch': best_epoch
        }
        with open('model/training_params.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        # Save training history
        if history:
            with open('model/training_history.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)

def main():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    special_tokens_dict = {'additional_special_tokens': ['[SUBJECT]', '[BODY]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
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
    trainer = BertTrainer()
    trainer.tokenizer = tokenizer  # Ensure trainer uses the tokenizer with special tokens
    trainer.model.resize_token_embeddings(len(tokenizer))
    
    # Train model
    trainer.train(train_dataset, val_dataset)

    # Save tokenizer with model
    trainer.tokenizer.save_pretrained('model/best_model')
    trainer.tokenizer.save_pretrained('model/latest_model')

if __name__ == "__main__":
    main() 