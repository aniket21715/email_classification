import pickle
import os
import logging
from typing import Union

logger = logging.getLogger(__name__)

# Try to import CatBoost, fallback to sklearn if not available
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    from sklearn.linear_model import LogisticRegression
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available, using LogisticRegression as fallback")

class EmailClassifier:
    def __init__(self, model_dir="models"):
        self.model = None
        self.vectorizer = None
        self.model_path = None
        self.vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

        # Find the model file
        cbm_path = os.path.join(model_dir, "classifier_model.cbm")
        joblib_path = os.path.join(model_dir, "email_classifier.joblib")
        pkl_path = os.path.join(model_dir, "classifier_model.pkl")

        if CATBOOST_AVAILABLE and os.path.exists(cbm_path):
            self.model_path = cbm_path
        elif os.path.exists(joblib_path):
            self.model_path = joblib_path
        elif os.path.exists(pkl_path):
            self.model_path = pkl_path
        else:
            logger.error("No valid model file found in models directory.")
            self.model_path = None

        self._load_model()

    def _load_model(self):
        """Load pre-trained model and vectorizer"""
        try:
            # Load vectorizer
            if os.path.exists(self.vectorizer_path):
                with open(self.vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Vectorizer loaded successfully")

            # Load model
            if self.model_path and os.path.exists(self.model_path):
                if CATBOOST_AVAILABLE and self.model_path.endswith('.cbm'):
                    self.model = CatBoostClassifier()
                    self.model.load_model(self.model_path)
                elif self.model_path.endswith('.joblib'):
                    import joblib
                    self.model = joblib.load(self.model_path)
                else:
                    with open(self.model_path, "rb") as f:
                        self.model = pickle.load(f)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def predict(self, text: str) -> str:
        """Predict email category"""
        if not self.model or not self.vectorizer:
            logger.warning("Model or vectorizer not loaded. Training a basic model...")
            return "uncategorized"
        
        try:
            # Vectorize input
            X = self.vectorizer.transform([text])
            
            # Predict
            if CATBOOST_AVAILABLE and hasattr(self.model, 'predict'):
                prediction = self.model.predict(X)
                return prediction[0] if isinstance(prediction, list) else prediction
            else:
                return self.model.predict(X)[0]
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "error"
    
    def get_prediction_probabilities(self, text: str) -> dict:
        """Get prediction probabilities for all classes"""
        if not self.model or not self.vectorizer:
            return {}
        
        try:
            X = self.vectorizer.transform([text])
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X)[0]
                classes = self.model.classes_ if hasattr(self.model, 'classes_') else []
                return dict(zip(classes, probas))
            return {}
        except Exception as e:
            logger.error(f"Error getting probabilities: {e}")
            return {}