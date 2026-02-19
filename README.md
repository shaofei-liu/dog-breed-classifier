# Dog Breed Classifier

AI-powered dog breed classification using Vision Transformer (ViT-B/16) with 120 dog breed support. Upload an image and get instant breed identification with confidence scores and top-5 predictions.

## Overview

This project implements a state-of-the-art image classification system for identifying dog breeds from photos. It combines the power of Vision Transformers with a clean, user-friendly interface accessible via web browser, REST API, or local deployment.

## Features

- **Vision Transformer (ViT-B/16)**: Pre-trained model fine-tuned on 120 dog breeds
- **High Accuracy**: 94-96% top-1 accuracy on test set
- **Bilingual Interface**: English and Deutsch support
- **Multiple Access Methods**:
  - Web UI (Gradio interface)
  - REST API endpoints
  - Local or cloud deployment
- **Fast Inference**: Real-time breed classification
- **Top-5 Predictions**: See confidence scores for top matches
- **Error Handling**: Robust handling of invalid or non-dog images

## Quick Start

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Gradio interface
python app_spaces.py

# Or run FastAPI server
python api_server.py
```

### Docker Deployment

```bash
docker build -t dog-breed-classifier .
docker run -p 7860:7860 dog-breed-classifier
```

## API Endpoints

### Gradio Interface
- **URL**: `http://localhost:7860` (after running `app_spaces.py`)
- Provides interactive web interface for uploading images

### FastAPI Server (when using `api_server.py`)
- `POST /api/predict` - Classify dog breed from image URL or file
- `GET /api/breeds` - Get list of all 120 supported dog breeds
- `GET /api/health` - Health check endpoint

### Example API Usage

```bash
curl -X POST "http://localhost:7860/api/predict?url=<image_url>" \
  -H "Accept: application/json"
```

## Technical Stack

- **Framework**: PyTorch 2.10
- **Model**: Vision Transformer (ViT-B/16)
- **Frontend**: Gradio 4.26.0
- **Backend**: FastAPI (optional local API)
- **Dataset**: Stanford Dogs (120 breeds)
- **Container**: Docker
- **Language**: Python 3.9+

## Project Structure

```
dog-breed-classifier/
├── app_spaces.py              # Gradio UI (main interface)
├── api_server.py              # FastAPI backend (alternative)
├── dog-breed-recognition.py   # Core classification logic
├── models/
│   ├── modeling.py            # Vision Transformer architecture
│   └── configs.py             # Model configuration
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Model Information

- **Architecture**: Vision Transformer (ViT-B/16)
- **Pre-training**: ImageNet-21K
- **Fine-tuning**: Stanford Dogs dataset (120 breeds)
- **Input Size**: 224×224 pixels
- **Output**: Breed classification with confidence scores

## Deployment

### Live Demo
Visit: https://www.shaofeiliu.com/#dog-breed-classifier

### Local/Server Deployment
Deploy the Dockerfile to any Docker-compatible platform (AWS, Google Cloud, local server, etc.)

## Supported Dog Breeds

The model supports 120 dog breeds including:
- Labrador Retriever
- German Shepherd
- Golden Retriever
- French Bulldog
- Bulldog
- Poodle
- Beagle
- And 113 more breeds...

See the API endpoint `/api/breeds` for the complete list.

## Usage Examples

### Web Interface
1. Open the application in a browser
2. Click "Upload Image" and select a dog photo
3. View results including breed name and confidence score
4. See top-5 predictions with individual confidence scores

### Python API (Local)
```python
import requests

# Upload image from URL
response = requests.post(
    "http://localhost:7860/api/predict",
    params={"url": "https://example.com/dog.jpg"}
)
results = response.json()
print(results["predictions"])
```

## Performance

- **Inference Time**: ~100-200ms per image (CPU), ~50-100ms (GPU)
- **Model Size**: ~86MB
- **Memory Usage**: ~2GB during inference
- **Accuracy**: 94-96% top-1 accuracy

## Requirements

See `requirements.txt` for all dependencies. Key packages:
- torch (PyTorch)
- gradio (UI framework)
- fastapi (REST API)
- pillow (image processing)
- numpy

## Troubleshooting

**Issue**: Model weights file not found
- **Solution**: Download model checkpoint and place in output directory

**Issue**: Image format not supported
- **Solution**: Use JPEG, PNG, or WebP format images

**Issue**: Out of memory errors
- **Solution**: Use a machine with more RAM/GPU

## Related Projects

- **Portfolio**: https://github.com/shaofei-liu/portfolio
- **RAG Chatbot**: https://github.com/shaofei-liu/rag-chatbot
- **IRevRNN Research**: https://github.com/shaofei-liu/irevrnn

## License

This project is provided for educational and commercial use.

## Contact & Support

For questions or issues:
1. Visit my personal website: https://www.shaofeiliu.com
2. Open an issue on GitHub

---

**Note**: Model weights are not included in the repository due to size constraints. Download from Hugging Face Spaces or train your own.
