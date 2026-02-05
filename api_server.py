"""
Dog Breed Classification API Server
FastAPI 服务器用于模型推理
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 FastAPI
app = FastAPI(
    title="Dog Breed Classification API",
    description="使用 Vision Transformer 识别犬种",
    version="1.0.0"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型参数
MODEL_TYPE = "ViT-B_16"
CHECKPOINT_PATH = "./output/sample_run_checkpoint.bin"
IMG_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 120个犬种类别
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

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 全局模型变量
model = None

def load_model():
    """加载模型"""
    global model
    try:
        logger.info(f"设备: {DEVICE}")
        logger.info(f"加载配置: {MODEL_TYPE}")
        
        config = CONFIGS[MODEL_TYPE]
        config.split = 'overlap'
        config.slide_step = 12
        
        logger.info("初始化模型 (120个犬种)...")
        model = VisionTransformer(config, IMG_SIZE, zero_head=True, num_classes=120)
        
        logger.info(f"加载检查点: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        
        model.to(DEVICE)
        model.eval()
        
        logger.info("✓ 模型加载成功")
        return True
    except Exception as e:
        logger.error(f"✗ 模型加载失败: {e}")
        return False

# 启动时加载模型
@app.on_event("startup")
async def startup_event():
    if not load_model():
        logger.warning("模型加载失败，API 将在演示模式运行")

@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_type": MODEL_TYPE,
        "num_classes": len(CLASS_LABELS)
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    预测犬种
    
    Args:
        file: 上传的图片文件
        
    Returns:
        json: 包含预测结果的字典
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 预处理
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # 推理
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)
        
        class_idx = class_idx.item()
        confidence = confidence.item()
        
        # 获取 Top 5 预测
        top_5_indices = torch.argsort(probs[0], descending=True)[:5]
        top_5_predictions = {
            CLASS_LABELS[idx.item()]: float(probs[0, idx].item())
            for idx in top_5_indices
        }
        
        logger.info(f"预测: {CLASS_LABELS[class_idx]} ({confidence:.2%})")
        
        return JSONResponse({
            "success": True,
            "breed": CLASS_LABELS[class_idx],
            "confidence": confidence,
            "top_5": top_5_predictions
        })
    
    except Exception as e:
        logger.error(f"预测出错: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/breeds")
async def get_breeds():
    """获取所有支持的犬种列表"""
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
