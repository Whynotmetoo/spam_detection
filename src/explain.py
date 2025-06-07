import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from typing import Dict, List, Tuple
import logging
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
            # Ensure text is not empty
            if not text or not text.strip():
                raise ValueError("Input text cannot be empty")
            
            # Log input text for debugging
            logger.info(f"Generating explanation for text: {text[:100]}...")
            
            # Get explanation
            try:
                exp = self.explainer.explain_instance(
                    text,
                    self.predict_proba,
                    num_features=self.num_features,
                    num_samples=self.num_samples,
                    top_labels=1  # Only explain the predicted class
                )
            except Exception as e:
                logger.error(f"LIME explanation failed: {str(e)}")
                raise ValueError(f"Failed to generate explanation: {str(e)}")
            
            # Get feature importance and filter out custom tokens
            feature_importance = {}
            try:
                for feature, score in exp.as_list():
                    # Skip any feature containing SUBJECT or BODY
                    if 'SUBJECT' in feature or 'BODY' in feature:
                        continue
                    # Skip stopwords and pure punctuation
                    feature_clean = feature.strip().lower().strip(string.punctuation)
                    if feature_clean in ENGLISH_STOP_WORDS or not feature_clean:
                        continue
                    feature_importance[feature] = score
                
                # Check if we have any features after filtering
                if not feature_importance:
                    logger.warning("No features found after filtering custom tokens")
                    raise ValueError("No meaningful features found in the explanation")
                
            except Exception as e:
                logger.error(f"Error processing features: {str(e)}")
                raise ValueError(f"Failed to process features: {str(e)}")
            
            # Create explanation plot
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get feature names and scores (excluding custom tokens)
                features = list(feature_importance.keys())
                scores = list(feature_importance.values())
                
                # Sort features by absolute importance
                sorted_indices = np.argsort(np.abs(scores))
                features = [features[i] for i in sorted_indices]
                scores = [scores[i] for i in sorted_indices]
                
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
                
            except Exception as e:
                logger.error(f"Error creating plot: {str(e)}")
                raise ValueError(f"Failed to create explanation plot: {str(e)}")
            
            return feature_importance, fig
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise 