import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import logging
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from dataset import load_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(
        self,
        model_dir: str = 'model/best_model',
        device: str = None
    ):
        """
        Initialize model evaluator.
        
        Args:
            model_dir (str): Directory containing saved model
            device (str): Device to use for evaluation
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load model and tokenizer
            if model_dir == 'bert-base-uncased':
                # For original BERT model
                self.model = BertForSequenceClassification.from_pretrained(
                    model_dir,
                    num_labels=2,
                    problem_type="single_label_classification"
                )
                self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            else:
                # For fine-tuned model
                if not os.path.exists(model_dir):
                    raise ValueError(f"Model directory {model_dir} does not exist")
                self.model = BertForSequenceClassification.from_pretrained(
                    model_dir,
                    num_labels=2,
                    problem_type="single_label_classification"
                )
                self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded model from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Set max length
        self.max_length = 512
    
    def predict_proba(self, subjects: list, bodies: list) -> np.ndarray:
        """
        Get probability predictions for texts.
        
        Args:
            subjects (list): List of email subjects
            bodies (list): List of email bodies
            
        Returns:
            np.ndarray: Probability predictions
        """
        # Combine subjects and bodies
        texts = [f"[SUBJECT] {subject} [BODY] {body}" for subject, body in zip(subjects, bodies)]
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            probs = torch.softmax(outputs.logits, dim=1)
        
        return probs.cpu().numpy()
    
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            Tuple[Dict[str, float], np.ndarray, np.ndarray]: Metrics, predictions, and true labels
        """
        all_preds = []
        all_probs = []
        all_labels = []
        
        try:
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Evaluating'):
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Get predictions and probabilities
                    probs = torch.softmax(outputs.logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of spam class
                    all_labels.extend(labels.cpu().numpy())
            
            # Convert to numpy arrays
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            
            # Compute metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels,
                all_preds,
                average='binary'
            )
            accuracy = accuracy_score(all_labels, all_preds)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            return metrics, all_preds, all_labels
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str = 'results/confusion_matrix.png'
    ) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            output_path (str): Path to save plot
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {output_path}")
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        output_path: str = 'results/roc_curve.png'
    ) -> None:
        """
        Plot and save ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            output_path (str): Path to save plot
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved ROC curve to {output_path}")

def main():
    # Load datasets
    _, _, test_dataset = load_datasets()
    logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate model
    metrics, preds, labels = evaluator.evaluate(test_loader)
    
    # Print metrics
    logger.info("\nTest Set Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric.capitalize()}: {value:.4f}")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(labels, preds)
    
    # Get probabilities for ROC curve
    probs = evaluator.predict_proba(test_dataset.subjects, test_dataset.bodies)
    evaluator.plot_roc_curve(labels, probs[:, 1])
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main() 