from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
import logging
import traceback
import os
from pii_masker import mask_pii
from email_classifier import EmailClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Email Classification System",
    description="Classifies support emails and masks PII",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory containing the static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize classifier
try:
    classifier = EmailClassifier()
    logger.info("Email classifier initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize classifier: {e}")
    classifier = None

class PiiEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailRequest(BaseModel):
    input_email_body: str
    
    @validator('input_email_body')
    def validate_email_body(cls, v):
        if not v or not v.strip():
            raise ValueError('Email body cannot be empty')
        if len(v) > 10000:  # Limit email size
            raise ValueError('Email body too long (max 10000 characters)')
        return v.strip()

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[PiiEntity]
    masked_email: str
    category_of_the_email: str
    prediction_probabilities: Optional[Dict[str, float]] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the index.html file"""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/classify", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    """
    Classify email and mask PII
    
    Returns:
    - input_email_body: Original email text
    - list_of_masked_entities: List of detected PII entities
    - masked_email: Email with PII masked
    - category_of_the_email: Predicted category
    - prediction_probabilities: Confidence scores for each category (optional)
    """
    try:
        text = request.input_email_body
        
        # Mask PII
        masked_text, entities = mask_pii(text)
        
        # Classify email
        if classifier:
            try:
                category = classifier.predict(masked_text)
                probabilities = classifier.get_prediction_probabilities(masked_text)
            except Exception as e:
                logger.error(f"Classification error: {e}")
                logger.error(traceback.format_exc())
                category = "error_during_classification"
                probabilities = {}
        else:
            category = "classifier_not_available"
            probabilities = {}
        
        return EmailResponse(
            input_email_body=text,
            list_of_masked_entities=entities,
            masked_email=masked_text,
            category_of_the_email=category,
            prediction_probabilities=probabilities
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in classify_email: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Returns current status of the service and classifier
    """
    return {
        "status": "healthy",
        "classifier_available": classifier is not None,
        "classifier_type": classifier.__class__.__name__ if classifier else None
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions"""
    logger.error(f"Global exception handler caught: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": str(type(exc).__name__)}
    )
