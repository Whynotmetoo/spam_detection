import os
import torch
import numpy as np
from typing import Dict, Tuple, Union
import logging
from email.parser import Parser
from bs4 import BeautifulSoup
import re
from transformers import BertForSequenceClassification, BertTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailPredictor:
    def __init__(
        self,
        model_dir: str = 'model/best_model',
        device: str = None
    ):
        """
        Initialize email predictor.
        
        Args:
            model_dir (str): Directory containing saved model
            device (str): Device to use for inference
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model from {model_dir}")
        
        # Initialize email parser
        self.email_parser = Parser()
        
        # Set max length
        self.max_length = 512
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize email text.
        
        Args:
            text (str): Raw email text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_email_content(self, email_path: str) -> Tuple[str, str]:
        """
        Extract subject and body from email file.
        
        Args:
            email_path (str): Path to email file
            
        Returns:
            Tuple[str, str]: Subject and body text
        """
        try:
            with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()
            
            email = self.email_parser.parsestr(email_content)
            
            # Extract subject
            subject = email.get('subject', '')
            if subject is None:
                subject = ''
            
            # Extract body
            body = ''
            if email.is_multipart():
                for part in email.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = email.get_payload(decode=True).decode()
            
            # Clean subject and body
            subject = self.clean_text(subject)
            body = self.clean_text(body)
            
            return subject, body
            
        except Exception as e:
            logger.error(f"Error processing email {email_path}: {str(e)}")
            return '', ''
    
    def predict(
        self,
        text: Union[str, list] = "",
        subject: Union[str, list] = "",
        threshold: float = 0.5
    ) -> Union[Dict[str, Union[str, float]], list]:
        """
        Predict if email(s) is/are spam.
        
        Args:
            text (Union[str, list]): Email text(s) or path(s) to email file(s)
            threshold (float): Classification threshold
            
        Returns:
            Union[Dict[str, Union[str, float]], list]: Prediction(s) with confidence
        """
        # Handle single text/file
        if isinstance(text, str):
            # Check if text is a file path
            if os.path.isfile(text):
                subject, body = self.extract_email_content(text)
            else:
                # If it's a single text, treat it as body
                subject = self.clean_text(subject)
                body = self.clean_text(text)
            
            subjects = [subject]
            bodies = [body]
            single_input = True
        else:
            # Handle multiple texts/files
            subjects = []
            bodies = []
            for t in text:
                if os.path.isfile(t):
                    subject, body = self.extract_email_content(t)
                else:
                    subject = ""
                    body = self.clean_text(t)
                subjects.append(subject)
                bodies.append(body)
            single_input = False
        
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
        
        # Convert to numpy
        probs = probs.cpu().numpy()
        
        # Process predictions
        predictions = []
        for prob in probs:
            spam_prob = prob[1]  # Probability of spam class
            is_spam = spam_prob >= threshold
            predictions.append({
                'prediction': 'spam' if is_spam else 'ham',
                'confidence': float(spam_prob if is_spam else 1 - spam_prob)
            })
        
        return predictions[0] if single_input else predictions

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict if email is spam')
    parser.add_argument(
        'input',
        help='Email text or path to email file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='model/best_model',
        help='Path to model directory (default: model/best_model)'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = EmailPredictor(model_dir=args.model_dir)
    
    # Make prediction
    result = predictor.predict(args.input, args.threshold)
    
    # Print result
    print(f"\nPrediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")

if __name__ == "__main__":
    main() 