import os
import pandas as pd
import re
from bs4 import BeautifulSoup
from email.parser import Parser
from typing import List, Tuple
import logging
import random
from pathlib import Path
import glob
import chardet
import email
from email.header import decode_header
from langdetect import detect, LangDetectException
import base64
import quopri

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailPreprocessor:
    def __init__(self):
        self.email_parser = Parser()
        self.encoding_errors = 0
        self.total_emails = 0
        self.non_english_emails = 0
        self.invalid_content_emails = 0
    
    def is_english(self, text: str) -> bool:
        """
        Check if the text is in English.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text is in English, False otherwise
        """
        if not text.strip():
            return False
            
        try:
            # Take first 1000 characters for language detection to improve speed
            sample = text[:1000]
            return detect(sample) == 'en'
        except LangDetectException:
            return False
    
    def detect_encoding(self, content: bytes) -> str:
        """
        Detect the encoding of the content.
        
        Args:
            content (bytes): Raw content
            
        Returns:
            str: Detected encoding
        """
        result = chardet.detect(content)
        return result['encoding'] or 'latin1'  # fallback to latin1 if detection fails
    
    def decode_email_header(self, header: str) -> str:
        """
        Decode email header.
        
        Args:
            header (str): Raw header string
            
        Returns:
            str: Decoded header
        """
        if not header:
            return ""
            
        decoded_parts = []
        for part, encoding in decode_header(header):
            if isinstance(part, bytes):
                try:
                    decoded_parts.append(part.decode(encoding or 'utf-8', errors='replace'))
                except Exception:
                    decoded_parts.append(part.decode('latin1', errors='replace'))
            else:
                decoded_parts.append(str(part))
        return ' '.join(decoded_parts)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize email text.
        
        Args:
            text (str): Raw email text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Check if text contains HTML-like content
        if '<' in text and '>' in text:
            try:
                # Only use BeautifulSoup if the text contains HTML-like content
                text = BeautifulSoup(text, 'html.parser').get_text()
            except Exception:
                # If HTML parsing fails, try to remove common HTML tags manually
                text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs but keep the domain names
        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses but keep the domain names
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_valid_content(self, text: str) -> bool:
        """
        Check if the content is valid and meaningful.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if content is valid, False otherwise
        """
        if not text.strip():
            return False
            
        # Check for common non-meaningful content patterns
        invalid_patterns = [
            r'content-transfer-encoding',
            r'content-type:',
            r'content-id:',
            r'image/gif',
            r'image/jpeg',
            r'image/png',
            r'application/',
            r'multipart/',
            r'boundary=',
            r'charset=',
            r'name=',
            r'filename=',
            r'base64',
            r'quoted-printable'
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, text.lower()):
                return False
                
        # Check if text is too short (less than 10 characters)
        if len(text.strip()) < 10:
            return False
            
        return True
    
    def decode_content(self, part) -> str:
        """
        Decode email part content based on its encoding.
        
        Args:
            part: Email part
            
        Returns:
            str: Decoded content
        """
        try:
            content_type = part.get_content_type()
            
            # Skip non-text content
            if not content_type.startswith('text/'):
                return ""
                
            # Get content and encoding
            payload = part.get_payload(decode=True)
            if not payload:
                return ""
                
            # Detect encoding
            encoding = self.detect_encoding(payload)
            
            # Decode content
            try:
                return payload.decode(encoding, errors='replace')
            except Exception:
                return payload.decode('latin1', errors='replace')
                
        except Exception:
            return ""
    
    def extract_email_content(self, email_content: str) -> Tuple[str, str]:
        """
        Extract subject and body from email content.
        
        Args:
            email_content (str): Raw email content
            
        Returns:
            Tuple[str, str]: Subject and body text
        """
        try:
            email = self.email_parser.parsestr(email_content)
            
            # Extract and decode subject
            subject = self.decode_email_header(email.get('subject', ''))
            
            # Extract body
            body_parts = []
            
            if email.is_multipart():
                for part in email.walk():
                    content = self.decode_content(part)
                    if content and self.is_valid_content(content):
                        body_parts.append(content)
            else:
                content = self.decode_content(email)
                if content and self.is_valid_content(content):
                    body_parts.append(content)
            
            body = ' '.join(body_parts)
            
            return subject, body
            
        except Exception as e:
            self.encoding_errors += 1
            return '', ''
    
    def process_enron_emails(self, csv_path: str, sample_size: int = 5000) -> pd.DataFrame:
        """
        Process Enron emails from CSV file.
        
        Args:
            csv_path (str): Path to Enron emails CSV file
            sample_size (int): Number of emails to sample
            
        Returns:
            pd.DataFrame: Processed emails with labels
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Process emails
            processed_data = []
            for _, row in df.iterrows():
                try:
                    subject, body = self.extract_email_content(row['message'])
                    
                    # Clean subject and body separately
                    cleaned_subject = self.clean_text(subject)
                    cleaned_body = self.clean_text(body)
                    
                    # Combine for language detection
                    full_text = f"{cleaned_subject} {cleaned_body}"
                    
                    if cleaned_subject.strip() or cleaned_body.strip():  # Only add non-empty texts
                        if self.is_valid_content(full_text) and self.is_english(full_text):
                            processed_data.append({
                                'subject': cleaned_subject,
                                'body': cleaned_body,
                                'label': 0  # Ham
                            })
                        else:
                            self.non_english_emails += 1
                            self.invalid_content_emails += 1
                            
                        # Stop if we have enough English emails
                        if len(processed_data) >= sample_size:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error processing Enron email {row['file']}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(processed_data)} valid English ham emails, skipped {self.non_english_emails} non-English emails and {self.invalid_content_emails} invalid content emails")
            return pd.DataFrame(processed_data)
            
        except Exception as e:
            logger.error(f"Error processing Enron emails: {str(e)}")
            return pd.DataFrame()
    
    def process_spam_emails(self, spam_dir: str, sample_size: int = 5000) -> pd.DataFrame:
        """
        Process spam emails from directory.
        
        Args:
            spam_dir (str): Directory containing spam emails
            sample_size (int): Number of emails to sample
            
        Returns:
            pd.DataFrame: Processed emails with labels
        """
        try:
            # Get all email files
            email_files = []
            for ext in ['*.lorien', '*.eml', '*.txt']:
                email_files.extend(glob.glob(os.path.join(spam_dir, '**', ext), recursive=True))
            
            # Shuffle files
            random.shuffle(email_files)
            
            # Process emails
            processed_data = []
            for email_path in email_files:
                try:
                    # Read file in binary mode
                    with open(email_path, 'rb') as f:
                        content = f.read()
                    
                    # Detect encoding
                    encoding = self.detect_encoding(content)
                    
                    # Decode content
                    email_content = content.decode(encoding, errors='replace')
                    
                    subject, body = self.extract_email_content(email_content)
                    
                    # Clean subject and body separately
                    cleaned_subject = self.clean_text(subject)
                    cleaned_body = self.clean_text(body)
                    
                    # Combine for language detection
                    full_text = f"{cleaned_subject} {cleaned_body}"
                    
                    if cleaned_subject.strip() or cleaned_body.strip():  # Only add non-empty texts
                        if self.is_valid_content(full_text) and self.is_english(full_text):
                            processed_data.append({
                                'subject': cleaned_subject,
                                'body': cleaned_body,
                                'label': 1  # Spam
                            })
                        else:
                            self.non_english_emails += 1
                            self.invalid_content_emails += 1
                            
                        # Stop if we have enough English emails
                        if len(processed_data) >= sample_size:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error processing spam email {email_path}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(processed_data)} valid English spam emails, skipped {self.non_english_emails} non-English emails and {self.invalid_content_emails} invalid content emails")
            return pd.DataFrame(processed_data)
            
        except Exception as e:
            logger.error(f"Error processing spam emails: {str(e)}")
            return pd.DataFrame()
    
    def split_dataset(self, df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            df (pd.DataFrame): Full dataset
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test sets
        """
        # Shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split dataset
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        return train_df, val_df, test_df

def main():
    # Clean datasets directory
    datasets_dir = 'datasets'
    if os.path.exists(datasets_dir):
        for file in os.listdir(datasets_dir):
            file_path = os.path.join(datasets_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {str(e)}")
    else:
        os.makedirs(datasets_dir)
    
    preprocessor = EmailPreprocessor()
    
    # Process Enron emails
    enron_df = preprocessor.process_enron_emails('data/Enron Email Dataset/emails.csv')
    logger.info(f"Processed {len(enron_df)} Enron emails")
    
    # Process spam emails
    spam_df = preprocessor.process_spam_emails('data/SPAM Archive Dataset')
    logger.info(f"Processed {len(spam_df)} spam emails")
    
    # Combine datasets
    combined_df = pd.concat([enron_df, spam_df], ignore_index=True)
    logger.info(f"Combined dataset has {len(combined_df)} total emails")
    
    # Split dataset
    train_df, val_df, test_df = preprocessor.split_dataset(combined_df)
    logger.info(f"Split dataset into {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test samples")
    
    # Save datasets
    train_df.to_csv('datasets/train.csv', index=False)
    val_df.to_csv('datasets/validate.csv', index=False)
    test_df.to_csv('datasets/test.csv', index=False)
    logger.info("Saved datasets to CSV files")

if __name__ == "__main__":
    main() 