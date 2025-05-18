import re
import logging
from typing import List, Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import spacy, handle if not available
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError) as e:
    logger.warning(f"SpaCy not available: {e}. Person detection will be disabled.")
    nlp = None

# Enhanced regex patterns with better validation
_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    "phone_number": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1,2}\d{3,4}[-.\s]?\d{3,4}\b'),
    "dob": re.compile(r'\b(?:0?[1-9]|[12][0-9]|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)?\d{2}\b'),
    "aadhar_num": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
    "credit_debit_no": re.compile(r'\b(?:\d{4}[-\s]?){3,4}\d{1,4}\b'),
    "cvv_no": re.compile(r'\b\d{3,4}\b(?=\s|$)'),
    "expiry_date": re.compile(r'\b(?:0[1-9]|1[0-2])[/-](?:\d{2}|\d{4})\b'),
    "ssn": re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
    "account_number": re.compile(r'\b(?:account|acct)[\s#:]*\d{8,17}\b', re.IGNORECASE)
}

def mask_pii(text: str) -> Tuple[str, List[Dict]]:
    """
    Detects and masks PII in text.
    
    Args:
        text: Input text to process
        
    Returns:
        Tuple of (masked_text, list_of_entities)
        Each entity: {"position": [start, end], "classification": label, "entity": original_text}
    """
    if not text or not isinstance(text, str):
        return "", []
    
    entities = []
    
    # 1) SpaCy PERSON detection (if available)
    if nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.strip()) > 1:
                    entities.append({
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "label": "full_name",
                        "text": ent.text
                    })
        except Exception as e:
            logger.warning(f"SpaCy processing failed: {e}")

    # 2) Regex-based detection
    for label, pattern in _PATTERNS.items():
        try:
            for match in pattern.finditer(text):
                # Additional validation for certain patterns
                if label == "cvv_no":
                    # Avoid matching random 3-4 digit numbers
                    context = text[max(0, match.start()-10):match.end()+10].lower()
                    if not any(word in context for word in ['cvv', 'cvc', 'security', 'code']):
                        continue
                
                entities.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": label,
                    "text": match.group()
                })
        except Exception as e:
            logger.warning(f"Pattern matching failed for {label}: {e}")

    # Sort by position and handle overlaps
    entities.sort(key=lambda e: e["start"])
    
    # Remove overlapping entities (keep the first one)
    filtered_entities = []
    last_end = 0
    for entity in entities:
        if entity["start"] >= last_end:
            filtered_entities.append(entity)
            last_end = entity["end"]

    # Build masked text
    masked_parts = []
    result_entities = []
    last_pos = 0
    
    for entity in filtered_entities:
        # Add text before entity
        masked_parts.append(text[last_pos:entity["start"]])
        # Add placeholder
        placeholder = f"[{entity['label'].upper()}]"
        masked_parts.append(placeholder)
        # Record entity for output
        result_entities.append({
            "position": [entity["start"], entity["end"]],
            "classification": entity["label"],
            "entity": entity["text"]
        })
        last_pos = entity["end"]
    
    # Add remaining text
    masked_parts.append(text[last_pos:])
    
    return "".join(masked_parts), result_entities