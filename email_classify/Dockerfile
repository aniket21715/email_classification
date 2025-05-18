FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_md

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models

# Train the model (this will create a model if it doesn't exist)
RUN python src/train_model.py

# Expose the port
EXPOSE 7860

# Command to run the application
CMD ["python", "-m", "uvicorn", "main:api", "--host", "0.0.0.0", "--port", "7860"]
