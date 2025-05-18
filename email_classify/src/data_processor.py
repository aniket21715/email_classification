import os
import pandas as pd
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

def extract_subject_body(email_text: str) -> tuple:
    """Extract subject and body from email text"""
    try:
        # Split on first newline after "Subject:"
        parts = email_text.split("Subject:", 1)
        if len(parts) > 1:
            # Further split the subject+body part
            subject_and_body = parts[1].strip().split("\n", 1)
            subject = subject_and_body[0].strip()
            body = subject_and_body[1].strip() if len(subject_and_body) > 1 else ""
            return subject, body
        return "", email_text
    except Exception as e:
        logger.error(f"Error extracting subject/body: {e}")
        return "", email_text

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    try:
        # Remove contact information patterns
        text = re.sub(r'You can reach me at.*?\.', '', text)
        text = re.sub(r'My name is.*?\.', '', text)
        text = re.sub(r'My contact number is.*?\.', '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove any remaining leading/trailing whitespace
        text = text.strip()
        
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text

def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess email data"""
    try:
        path = Path(path)
        if not path.exists():
            logger.warning(f"Data file not found: {path}")
            return pd.DataFrame()

        # Load CSV
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")

        # Extract subject and body from email column
        df['subject'] = df['email'].apply(lambda x: extract_subject_body(x)[0])
        df['body'] = df['email'].apply(lambda x: extract_subject_body(x)[1])
        
        # Clean the text content
        df['cleaned_body'] = df['body'].apply(clean_text)
        
        # Combine subject and cleaned body for classification
        df['text'] = df.apply(lambda x: f"{x['subject']} {x['cleaned_body']}", axis=1)
        
        # Standardize column names
        column_mapping = {
            'email_body': 'text',
            'category': 'label',
            'content': 'text',
            'class': 'label',
            'type': 'label'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Ensure required columns exist
        required_cols = ['text', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        # Clean data
        df = df.dropna(subset=required_cols)
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(str)

        # Remove empty texts
        df = df[df['text'].str.strip() != '']

        logger.info(f"After cleaning: {len(df)} rows, {df['label'].nunique()} categories")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()
