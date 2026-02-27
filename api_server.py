"""
Dog Breed Classification API Server
FastAPI server for model inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from models.modeling import VisionTransformer, CONFIGS
from PIL import Image
from torchvision import transforms
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Dog Breed Classification API",
    description="Dog breed classification using Vision Transformer",
    version="1.0.0"
)

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model parameters
MODEL_TYPE = "ViT-B_16"
CHECKPOINT_PATH = "./output/sample_run_checkpoint.bin"
IMG_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 120 dog breed categories
CLASS_LABELS = [
    "Chihuahua", "Japanese_Spaniel", "Maltese", "Pekingese", "Shih_Tzu",
    "Blenheim_Spaniel", "Papillon", "Toy_Terrier", "Rhodesian_Ridgeback", "Afghan_Hound",
    "Basset", "Beagle", "Bloodhound", "Bluetick", "Black_And_Tan_Coonhound",
    "Walker_Hound", "English_Foxhound", "Redbone", "Borzoi", "Irish_Wolfhound",
    "Italian_Greyhound", "Whippet", "Ibizan_Hound", "Norwegian_Elkhound", "Otterhound",
    "Saluki", "Scottish_Deerhound", "Weimaraner", "Staffordshire_Bullterrier", "American_Staffordshire_Terrier",
    "Bedlington_Terrier", "Border_Terrier", "Kerry_Blue_Terrier", "Irish_Terrier", "Norfolk_Terrier",
    "Norwich_Terrier", "Yorkshire_Terrier", "Wire_Haired_Fox_Terrier", "Lakeland_Terrier", "Sealyham_Terrier",
    "Airedale", "Cairn", "Australian_Terrier", "Dandie_Dinmont", "Boston_Bull",
    "Miniature_Schnauzer", "Giant_Schnauzer", "Standard_Schnauzer", "Scotch_Terrier", "Tibetan_Terrier",
    "Silky_Terrier", "Soft_Coated_Wheaten_Terrier", "West_Highland_White_Terrier", "Lhasa", "Flat_Coated_Retriever",
    "Curly_Coated_Retriever", "Golden_Retriever", "Labrador_Retriever", "Chesapeake_Bay_Retriever", "German_Short_Haired_Pointer",
    "Vizsla", "English_Setter", "Irish_Setter", "Gordon_Setter", "Brittany_Spaniel",
    "Clumber", "English_Springer", "Welsh_Springer_Spaniel", "Cocker_Spaniel", "Sussex_Spaniel",
    "Irish_Water_Spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois",
    "Briard", "Australian_Kelpie", "Komondor", "Old_English_Sheepdog", "Shetland_Sheepdog",
    "Collie", "Border_Collie", "Bouvier_Des_Flandres", "Rottweiler", "German_Shepherd",
    "Doberman", "Miniature_Pinscher", "Greater_Swiss_Mountain_Dog", "Bernese_Mountain_Dog", "Appenzeller",
    "Entlebucher", "Boxer", "Bull_Mastiff", "Tibetan_Mastiff", "French_Bulldog",
    "Great_Dane", "Saint_Bernard", "Eskimo_Dog", "Alaskan_Malamute", "Siberian_Husky",
    "Affenpinscher", "Basenji", "Pug", "Leonberger", "Newfoundland",
    "Great_Pyrenees", "Samoyed", "Pomeranian", "Chow", "Keeshond",
    "Brabancon_Griffon", "Pembroke_Welsh_Corgi", "Cardigan_Welsh_Corgi", "Toy_Poodle", "Miniature_Poodle",
    "Standard_Poodle", "Mexican_Hairless", "Dingo", "Dhole", "African_Hunting_Dog"
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global model variable
model = None

def load_model():
    """加载模型"""
    global model
    try:
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Loading config: {MODEL_TYPE}")
        
        config = CONFIGS[MODEL_TYPE]
        config.split = 'overlap'
        config.slide_step = 12
        
        logger.info("Initializing model (120 dog breeds)...")
        model = VisionTransformer(config, IMG_SIZE, zero_head=True, num_classes=120)
        
        logger.info(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        
        model.to(DEVICE)
        model.eval()
        
        logger.info("✓ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    if not load_model():
        logger.warning("Model loading failed, API will run in demo mode")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_type": MODEL_TYPE,
        "num_classes": len(CLASS_LABELS)
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(None), url: str = None):
    """
    Predict dog breed from file or URL
    
    Args:
        file: Image file to upload (optional)
        url: Image URL (optional - for webpage image extraction)
        
    Returns:
        json: Dictionary containing prediction results or image list for webpages
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        image = None
        
        # Handle file upload
        if file and file.filename:
            try:
                # Validate file type
                if file.content_type and "image" not in file.content_type.lower():
                    raise HTTPException(status_code=400, detail="Please upload an image file (JPEG, PNG, GIF, WebP, etc.)")
                
                logger.info(f"Processing file upload: {file.filename}")
                contents = await file.read()
                logger.info(f"File size: {len(contents)} bytes")
                if not contents:
                    raise HTTPException(status_code=400, detail="File is empty. Please upload an image file.")
                
                try:
                    image = Image.open(io.BytesIO(contents)).convert("RGB")
                    logger.info(f"Image loaded: {image.size}")
                except Exception as e:
                    error_str = str(e).lower()
                    logger.error(f"Failed to parse uploaded image: {e}")
                    if "cannot identify" in error_str or "unidentified" in error_str:
                        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image. Please upload a JPEG, PNG, GIF, or WebP file.")
                    else:
                        raise HTTPException(status_code=400, detail=f"Could not process image: {str(e)}")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Failed to process image file: {str(e)}")
        
        # Handle URL
        elif url:
            logger.info(f"Processing URL: {url}")
            import requests
            from bs4 import BeautifulSoup
            
            try:
                # Try to fetch from URL
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                
                # Check if it's an HTML page
                if 'text/html' in response.headers.get('Content-Type', ''):
                    # Extract images from webpage
                    soup = BeautifulSoup(response.content, 'html.parser')
                    images = []
                    
                    for img in soup.find_all('img'):
                        img_src = img.get('src') or img.get('data-src')
                        if img_src:
                            # Handle relative URLs
                            if img_src.startswith('http'):
                                absolute_url = img_src
                            elif img_src.startswith('/'):
                                from urllib.parse import urljoin
                                absolute_url = urljoin(url, img_src)
                            else:
                                continue
                            
                            images.append({
                                'url': absolute_url,
                                'alt': img.get('alt', 'Image')
                            })
                    
                    if images:
                        logger.info(f"Found {len(images)} images in webpage")
                        return JSONResponse({
                            "type": "image_selection",
                            "images": images[:20]  # Limit to 20 images
                        })
                    else:
                        raise HTTPException(status_code=400, detail="No images found in webpage")
                else:
                    # It's a direct image URL
                    if not response.content:
                        raise HTTPException(status_code=400, detail="Empty response from image URL")
                    try:
                        image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    except Exception as e:
                        error_str = str(e).lower()
                        if "cannot identify" in error_str or "unidentified" in error_str:
                            raise HTTPException(status_code=400, detail="The file at this URL is not a valid image. Please provide a direct link to a JPEG, PNG, GIF, or WebP image file.")
                        else:
                            raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {str(e)}")
            except HTTPException:
                raise
            except requests.RequestException as e:
                logger.error(f"Error fetching URL: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing URL image: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Failed to process URL image: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Either file or url parameter is required")
        
        # Predict if we have an image
        if image:
            logger.info("Preprocessing image...")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            logger.info("Running inference...")
            with torch.no_grad():
                logits = model(image_tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, class_idx = torch.max(probs, dim=1)
            
            class_idx = class_idx.item()
            confidence = confidence.item()
            
            # Get Top 5 predictions
            top_5_indices = torch.argsort(probs[0], descending=True)[:5]
            top_5_predictions = {
                CLASS_LABELS[idx.item()]: float(probs[0, idx].item())
                for idx in top_5_indices
            }
            
            logger.info(f"Prediction: {CLASS_LABELS[class_idx]} ({confidence:.2%})")
            
            return JSONResponse({
                "success": True,
                "breed": CLASS_LABELS[class_idx],
                "confidence": confidence,
                "top_5": top_5_predictions
            })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.post("/api/predict-selected-image")
async def predict_selected_image(image_url: str):
    """
    Predict dog breed from a specific selected image URL
    
    Args:
        image_url: URL of the image to predict
        
    Returns:
        json: Dictionary containing prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Processing selected image: {image_url}")
        
        import requests
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        logger.info("Preprocessing image...")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        logger.info("Running inference...")
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)
        
        class_idx = class_idx.item()
        confidence = confidence.item()
        
        # Get Top 5 predictions
        top_5_indices = torch.argsort(probs[0], descending=True)[:5]
        top_5_predictions = {
            CLASS_LABELS[idx.item()]: float(probs[0, idx].item())
            for idx in top_5_indices
        }
        
        logger.info(f"Prediction: {CLASS_LABELS[class_idx]} ({confidence:.2%})")
        
        return JSONResponse({
            "success": True,
            "breed": CLASS_LABELS[class_idx],
            "confidence": confidence,
            "top_5": top_5_predictions
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.get("/api/breeds")
async def get_breeds():
    """Get all supported dog breed list"""
    return {
        "total": len(CLASS_LABELS),
        "breeds": CLASS_LABELS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
