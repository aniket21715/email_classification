# Email Classification System for Support Team

This project implements an email classification system that categorizes support emails into predefined categories while masking personally identifiable information (PII) before processing.

## Features

- **Email Classification**: Categorizes support emails into Incident, Request, Change, and Problem categories
- **PII Masking**: Detects and masks personal information without using LLMs
- **API Deployment**: Exposes the solution via a FastAPI endpoint

## Architecture

The system consists of the following components:

1. **PII Masker**: Uses regex patterns to detect and mask personal information
2. **Email Classifier**: Uses TF-IDF vectorization with a Random Forest classifier
3. **API Server**: Provides endpoints for email classification and PII handling
4. **Data Processor**: Handles data loading, preparation, and preprocessing

## Setup Instructions

### Prerequisites

- Python 3.8+
- 8GB RAM (system designed for memory efficiency)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/email_classify.git
   cd email_classify
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Training the Model

To train the model using the provided or your own dataset:

```
python src/train_model.py --data_file your_data.csv --text_column email_text --label_column category
```

By default, if no data file is specified, the script will generate sample data for demonstration purposes.

### Running the API Server

Start the API server locally:

```
python main.py
```

The API will be accessible at `http://localhost:8000`.

## API Usage

### Classify Email Endpoint

**Endpoint**: `/classify`  
**Method**: POST  
**Request Body**:
```json
{
  "email_text": "Hello, my name is John Doe and I'm experiencing problems with my account. My email is john.doe@example.com."
}
```

**Response**:
```json
{
  "input_email_body": "Hello, my name is John Doe and I'm experiencing problems with my account. My email is john.doe@example.com.",
  "list_of_masked_entities": [
    {
      "position": [18, 26],
      "classification": "full_name",
      "entity": "John Doe"
    },
    {
      "position": [78, 97],
      "classification": "email",
      "entity": "john.doe@example.com"
    }
  ],
  "masked_email": "Hello, my name is [full_name] and I'm experiencing problems with my account. My email is [email].",
  "demasked_email": "Hello, my name is John Doe and I'm experiencing problems with my account. My email is john.doe@example.com.",
  "category": "Problem",
  "confidence": 0.85
}
```


## Directory Structure

```
email_classification_system/
├── data/
│   ├── raw/             # Raw data files
│   └── processed/       # Processed data with masked PII
├── models/              # Trained models
├── src/
│   ├── pii_masker.py    # PII detection and masking
│   ├── email_classifier.py # Email classification model
│   ├── data_processor.py # Data processing utilities
│   ├── train_model.py   # Training script
│   └── app.py           # API server
├── main.py               # Main application entry point for Hugging Face Spaces
├── Dockerfile           # Docker configuration for deployment
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## PII Types Detected

The system detects and masks the following PII types:

- Full Name ("full_name")
- Email Address ("email")
- Phone number ("phone_number")
- Date of birth ("dob")
- Aadhar card number ("aadhar_num")
- Credit/Debit Card Number ("credit_debit_no")
- CVV number ("cvv_no")
- Card expiry number ("expiry_no")

## Performance Considerations

- The system is designed to run efficiently on machines with limited resources (8GB RAM)
- The Random Forest classifier balances accuracy and resource usage
- PII masking uses optimized regex patterns to minimize processing time

## Troubleshooting

If you encounter any issues during deployment:

1. Check the Hugging Face Space logs to see any error messages
2. Ensure all dependencies are correctly listed in requirements.txt
3. Verify the correct ports are being used (default is 7860 for Hugging Face Spaces)
4. Confirm the model is properly loaded and initialized on startup
