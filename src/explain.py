import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from typing import Dict, List, Tuple
import logging
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailExplainer:
    def __init__(
        self,
        model,
        tokenizer: BertTokenizer,
        device: str = None,
        num_features: int = 10,
        num_samples: int = 1000
    ):
        """
        Initialize email explainer.
        
        Args:
            model: BERT model for prediction
            tokenizer (BertTokenizer): BERT tokenizer
            device (str): Device to use for inference
            num_features (int): Number of features to explain
            num_samples (int): Number of samples for LIME
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_samples = num_samples
        self.explainer = LimeTextExplainer(class_names=['ham', 'spam'])
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get probability predictions for texts.
        
        Args:
            texts (List[str]): List of texts
            
        Returns:
            np.ndarray: Probability predictions
        """
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=512,
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
    
    def explain_prediction(
        self,
        text: str,
        prediction: str,
        confidence: float
    ) -> Tuple[Dict[str, float], plt.Figure]:
        """
        Explain model prediction using LIME.
        
        Args:
            text (str): Input text
            prediction (str): Model prediction ('ham' or 'spam')
            confidence (float): Prediction confidence
            
        Returns:
            Tuple[Dict[str, float], plt.Figure]: Feature importance and explanation plot
        """
        try:
            # Get explanation
            exp = self.explainer.explain_instance(
                text,
                self.predict_proba,
                num_features=self.num_features,
                num_samples=self.num_samples
            )
            
            # Get feature importance
            feature_importance = dict(exp.as_list())
            
            # Create explanation plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get feature names and scores
            features = [f[0] for f in exp.as_list()]
            scores = [f[1] for f in exp.as_list()]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            colors = ['red' if s > 0 else 'green' for s in scores]
            
            ax.barh(y_pos, scores, align='center', color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # labels read top-to-bottom
            
            # Add labels and title
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'LIME Explanation for {prediction.upper()} Prediction (Confidence: {confidence:.1%})')
            
            # Add vertical line at x=0
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            
            return feature_importance, fig
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise 