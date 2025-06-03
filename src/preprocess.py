import os
import pandas as pd
import re
from bs4 import BeautifulSoup
from email.parser import Parser
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailPreprocessor:
    def __init__(self):
        self.email_parser = Parser()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize email text.
        
        Args:
            text (str): Raw email text
            
        Returns:
            str: Cleaned text
        """
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
            
            return subject, body
            
        except Exception as e:
            logger.error(f"Error processing email {email_path}: {str(e)}")
            return '', ''
    
    def process_email_files(self, input_dir: str, output_file: str, label: int) -> None:
        """
        Process all email files in a directory and save to CSV.
        
        Args:
            input_dir (str): Directory containing email files
            output_file (str): Path to output CSV file
            label (int): Label for emails in this directory (0 for ham, 1 for spam)
        """
        processed_data = []
        
        for filename in os.listdir(input_dir):
            if filename.endswith(('.eml', '.txt')):
                email_path = os.path.join(input_dir, filename)
                subject, body = self.extract_email_content(email_path)
                
                # Combine subject and body
                full_text = f"{subject} {body}"
                cleaned_text = self.clean_text(full_text)
                
                if cleaned_text.strip():  # Only add non-empty texts
                    processed_data.append({
                        'text': cleaned_text,
                        'label': label
                    })
        
        # Save to CSV
        df = pd.DataFrame(processed_data)
        df.to_csv(output_file, index=False)
        logger.info(f"Processed {len(processed_data)} emails from {input_dir}")

def main():
    preprocessor = EmailPreprocessor()
    
    # Process ham emails
    ham_dir = os.path.join('data', 'raw', 'ham')
    if os.path.exists(ham_dir):
        preprocessor.process_email_files(
            ham_dir,
            os.path.join('data', 'processed_ham.csv'),
            label=0
        )
    
    # Process spam emails
    spam_dir = os.path.join('data', 'raw', 'spam')
    if os.path.exists(spam_dir):
        preprocessor.process_email_files(
            spam_dir,
            os.path.join('data', 'processed_spam.csv'),
            label=1
        )
    
    # Combine datasets
    ham_df = pd.read_csv(os.path.join('data', 'processed_ham.csv')) if os.path.exists(os.path.join('data', 'processed_ham.csv')) else pd.DataFrame()
    spam_df = pd.read_csv(os.path.join('data', 'processed_spam.csv')) if os.path.exists(os.path.join('data', 'processed_spam.csv')) else pd.DataFrame()
    
    combined_df = pd.concat([ham_df, spam_df], ignore_index=True)
    combined_df.to_csv(os.path.join('data', 'processed.csv'), index=False)
    logger.info(f"Combined dataset saved with {len(combined_df)} total emails")

if __name__ == "__main__":
    main() 