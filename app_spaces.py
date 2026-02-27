import torch
import os
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from models.modeling import VisionTransformer, CONFIGS
from torchvision import transforms
from pathlib import Path
import io
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_TYPE = "ViT-B_16"
IMG_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

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

os.makedirs("./output", exist_ok=True)
CHECKPOINT_PATH = "./output/sample_run_checkpoint.bin"

logger.info("Loading model configuration...")
config = CONFIGS[MODEL_TYPE]
config.split = "overlap"
config.slide_step = 12
model = VisionTransformer(config, IMG_SIZE, zero_head=True, num_classes=120)

MODEL_LOADED = False
if Path(CHECKPOINT_PATH).exists():
    logger.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Model checkpoint loaded successfully")
        MODEL_LOADED = True
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        MODEL_LOADED = False
else:
    logger.warning(f"Checkpoint not found at {CHECKPOINT_PATH}")
    MODEL_LOADED = False

model.to(DEVICE)
model.eval()
logger.info("Model ready for inference")

transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_breed(image):
    if not MODEL_LOADED:
        return None, None, None
    
    if image is None:
        return None, None, None

    try:
        # Handle both numpy arrays and PIL Images
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            logger.error(f"Unexpected image type: {type(image)}")
            return None, None, None
            
        img = img.convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)

        class_idx = class_idx.item()
        confidence = confidence.item()

        top_5_indices = torch.argsort(probs[0], descending=True)[:5]
        top_5_dict = {}
        for idx in top_5_indices:
            idx_val = idx.item()
            conf = probs[0, idx_val].item()
            top_5_dict[CLASS_LABELS[idx_val]] = float(conf) * 100

        result = {
            "breed": CLASS_LABELS[class_idx],
            "confidence": float(confidence) * 100,
            "top_5": top_5_dict
        }

        return result, img, CLASS_LABELS[class_idx]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, None, None

def load_image_robust(image_bytes):
    """从字节流加载图像"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img

def extract_images_from_webpage(webpage_url):
    """从网页中提取所有图片 URLs
    
    Returns:
        list: 找到的图片 URLs 列表，最多返回前 10 个
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(webpage_url, timeout=15, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        image_urls = []
        
        # 提取所有 img 标签中的 src
        for img_tag in soup.find_all('img'):
            img_src = img_tag.get('src')
            if img_src:
                # 处理相对 URLs
                absolute_url = urljoin(webpage_url, img_src)
                image_urls.append(absolute_url)
        
        # 也尝试从 picture 标签中提取
        for picture_tag in soup.find_all('picture'):
            for source_tag in picture_tag.find_all('source'):
                srcset = source_tag.get('srcset')
                if srcset:
                    # srcset 可能包含多个 URLs，取第一个
                    img_url = srcset.split(',')[0].split()[0]
                    if img_url:
                        absolute_url = urljoin(webpage_url, img_url)
                        if absolute_url not in image_urls:
                            image_urls.append(absolute_url)
        
        # 过滤掉很小的图片（可能是图标）和重复的 URLs
        filtered_urls = []
        seen = set()
        for url in image_urls:
            if url not in seen:
                seen.add(url)
                filtered_urls.append(url)
        
        return filtered_urls[:10]  # 最多返回 10 个
    except Exception as e:
        logger.error(f"Error extracting images from webpage: {e}")
        return []

def is_webpage_url(url):
    """检查 URL 是否为网页而不是直接图片"""
    try:
        parsed = urlparse(url)
        # 获取路径和查询字符串
        path = parsed.path.lower()
        
        # 如果以常见图片扩展名结尾，则是直接图片
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg')
        if path.endswith(image_extensions):
            return False
        
        return True
    except:
        return True

# FastAPI 应用
app = FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境应该限制）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Dog Breed Classification API", "model_loaded": MODEL_LOADED, "timestamp": "2026-02-05T2"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(None), url: str = Query(None)):
    """预测犬种 - 返回前端期望的格式
    支持三种输入方式：
    1. 文件上传 (multipart/form-data with file)
    2. 直接图像 URL (url parameter - 直接指向图片文件)
    3. 网页 URL (url parameter - 自动提取网页中的图片)
    
    Updated Feb 27, 2026: Improved file upload handling with optional parameters
    """
    if not MODEL_LOADED:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Model not loaded. Please upload model checkpoint."}
        )
    
    try:
        image = None
        image_url = None  # 用于记录最终使用的图片 URL
        
        # 从文件上传加载图像
        if file:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 从 URL 加载图像
        elif url:
            # 检查是否为网页 URL
            if is_webpage_url(url):
                # 这是一个网页 URL，尝试提取图片
                image_urls = extract_images_from_webpage(url)
                
                if not image_urls:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": "No images found on the webpage. Please provide a webpage with dog images or use a direct image URL."}
                    )
                
                # 如果找到多个图片，返回列表供前端选择
                if len(image_urls) > 1:
                    return JSONResponse(
                        status_code=200,
                        content={
                            "success": True,
                            "type": "image_selection",
                            "images": image_urls,
                            "message": f"Found {len(image_urls)} images on the webpage. Please select one or use 'first' parameter to auto-select the first image."
                        }
                    )
                
                # 如果只找到一个图片，自动使用
                image_url = image_urls[0]
            else:
                # 这是一个直接图片 URL
                image_url = url
            
            # 加载图片 URL 对应的图像
            if image_url:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Accept": "image/*"
                    }
                    response = requests.get(image_url, timeout=10, headers=headers, allow_redirects=True)
                    response.raise_for_status()
                    
                    # 检查返回的内容是否为图片
                    content_type = response.headers.get('content-type', '')
                    if 'image' not in content_type.lower():
                        return JSONResponse(
                            status_code=400,
                            content={"success": False, "error": "No image found at this URL. Please provide a direct link to an image file."}
                        )
                    
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                except requests.exceptions.RequestException as e:
                    error_msg = str(e)
                    if "403" in error_msg:
                        error_msg = "Image URL is blocked. Please use a different image."
                    elif "404" in error_msg:
                        error_msg = "Image URL not found. Please check the URL is correct."
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": error_msg}
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    if "image" in error_str or "decode" in error_str or "format" in error_str:
                        return JSONResponse(
                            status_code=400,
                            content={"success": False, "error": "No image found at this URL. Please provide a valid image URL."}
                        )
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": f"Invalid image file: {str(e)}"}
                    )
        
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No file or URL provided"}
            )
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Failed to load image"}
            )
        
        result, _, breed = predict_breed(image)
        
        if result is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Prediction failed"}
            )
        
        # 返回前端期望的格式，包括 success 字段
        return JSONResponse(content={
            "success": True,
            "breed": result["breed"],
            "confidence": result["confidence"],
            "top_5": result["top_5"]
        })
    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/predict-selected-image")
async def predict_selected_image(image_url: str = Query(...)):
    """预测指定 URL 图片的犬种（用于网页图片选择）"""
    if not MODEL_LOADED:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Model not loaded"}
        )
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(image_url, timeout=10, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        result, _, breed = predict_breed(image)
        
        if result is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Prediction failed"}
            )
        
        return JSONResponse(content={
            "success": True,
            "breed": result["breed"],
            "confidence": result["confidence"],
            "top_5": result["top_5"]
        })
    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)