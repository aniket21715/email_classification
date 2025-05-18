import uvicorn
import logging
from src.app import app as api  # Expose FastAPI app as 'api' for Hugging Face Spaces

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    uvicorn.run(
        api,        
        host="localhost", 
        port=7860,
        log_level="info"
    )
