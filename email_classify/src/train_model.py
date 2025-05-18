import pickle
import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.data_processor import load_data
from src.pii_masker import mask_pii

# Try to import CatBoost, fallback to sklearn
try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    from sklearn.linear_model import LogisticRegression
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

def train_model(data_path="data/raw/combined_emails_with_natural_pii.csv", model_dir="models"):
    """Train email classification model"""
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    if df.empty:
        logger.error("Failed to load training data")
        return False
    
    # Mask PII in texts
    logger.info("Masking PII in training data...")
    masked_texts = []
    for text in df["text"]:
        masked_text, _ = mask_pii(text)
        masked_texts.append(masked_text)
    
    labels = df["label"].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        masked_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Vectorize
    logger.info("Vectorizing text data...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    logger.info("Training model...")
    if CATBOOST_AVAILABLE:
        model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            loss_function='MultiClass',
            verbose=False,
            random_seed=42
        )
        model.fit(
            Pool(X_train_vec, y_train),
            eval_set=Pool(X_test_vec, y_test),
            early_stopping_rounds=20,
            verbose=False
        )
        # Save CatBoost model
        model_path = os.path.join(model_dir, "classifier_model.cbm")
        model.save_model(model_path)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        # Save sklearn model
        model_path = os.path.join(model_dir, "classifier_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model accuracy: {accuracy:.3f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Vectorizer saved to {vectorizer_path}")
    return True

if __name__ == "__main__":
    train_model()