import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple, List
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaselineModels:
    def __init__(self):
        """
        Initialize baseline models with TF-IDF vectorizer.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.lr_model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42
        )
        self.nb_model = MultinomialNB(alpha=1.0)
        
    def prepare_text(self, subjects: List[str], bodies: List[str]) -> List[str]:
        """
        Combine subject and body for text processing.
        
        Args:
            subjects (List[str]): List of email subjects
            bodies (List[str]): List of email bodies
            
        Returns:
            List[str]: Combined texts
        """
        return [f"{subject} {body}" for subject, body in zip(subjects, bodies)]
    
    def train(self, train_df: pd.DataFrame) -> None:
        """
        Train both baseline models.
        
        Args:
            train_df (pd.DataFrame): Training data
        """
        # Prepare text data
        train_texts = self.prepare_text(train_df['subject'].values, train_df['body'].values)
        
        # Fit and transform text data
        X_train = self.vectorizer.fit_transform(train_texts)
        y_train = train_df['label'].values
        
        # Train Logistic Regression
        logger.info("Training Logistic Regression model...")
        self.lr_model.fit(X_train, y_train)
        
        # Train Naive Bayes
        logger.info("Training Naive Bayes model...")
        self.nb_model.fit(X_train, y_train)
        
        logger.info("Training completed!")
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        output_dir: str = 'results/baseline'
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate both models on test data.
        
        Args:
            test_df (pd.DataFrame): Test data
            output_dir (str): Directory to save results
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for both models
        """
        # Prepare text data
        test_texts = self.prepare_text(test_df['subject'].values, test_df['body'].values)
        X_test = self.vectorizer.transform(test_texts)
        y_test = test_df['label'].values
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Evaluate both models
        for model_name, model in [('Logistic Regression', self.lr_model), ('Naive Bayes', self.nb_model)]:
            # Get predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Compute metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            accuracy = accuracy_score(y_test, y_pred)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            results[model_name] = metrics
            
            # Plot confusion matrix
            self.plot_confusion_matrix(
                y_test, y_pred,
                output_path=os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
            )
            
            # Plot ROC curve
            self.plot_roc_curve(
                y_test, y_prob,
                output_path=os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_roc_curve.png')
            )
            
            # Print metrics
            logger.info(f"\n{model_name} Test Set Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric.capitalize()}: {value:.4f}")
        
        return results
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str
    ) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            output_path (str): Path to save plot
        """
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
        output_path: str
    ) -> None:
        """
        Plot and save ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            output_path (str): Path to save plot
        """
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
    
    def save_models(self, output_dir: str = 'model/baseline') -> None:
        """
        Save trained models and vectorizer.
        
        Args:
            output_dir (str): Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vectorizer
        joblib.dump(self.vectorizer, os.path.join(output_dir, 'vectorizer.joblib'))
        
        # Save models
        joblib.dump(self.lr_model, os.path.join(output_dir, 'logistic_regression.joblib'))
        joblib.dump(self.nb_model, os.path.join(output_dir, 'naive_bayes.joblib'))
        
        logger.info(f"Saved models to {output_dir}")
    
    def load_models(self, model_dir: str = 'model/baseline') -> None:
        """
        Load trained models and vectorizer.
        
        Args:
            model_dir (str): Directory containing saved models
        """
        # Load vectorizer
        self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
        
        # Load models
        self.lr_model = joblib.load(os.path.join(model_dir, 'logistic_regression.joblib'))
        self.nb_model = joblib.load(os.path.join(model_dir, 'naive_bayes.joblib'))
        
        logger.info(f"Loaded models from {model_dir}")

def main():
    # Load datasets
    train_df = pd.read_csv('datasets/train.csv')
    test_df = pd.read_csv('datasets/test.csv')
    
    logger.info(f"Loaded {len(train_df)} training and {len(test_df)} test samples")
    
    # Initialize and train models
    baseline = BaselineModels()
    baseline.train(train_df)
    
    # Evaluate models
    results = baseline.evaluate(test_df)
    
    # Save models
    baseline.save_models()
    
    logger.info("Baseline model evaluation completed!")

if __name__ == '__main__':
    main() 