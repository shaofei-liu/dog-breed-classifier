import torch
import numpy as np
from models.modeling import VisionTransformer, CONFIGS
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button

# 设定参数
model_type = "ViT-B_16"
checkpoint_path = "./output/sample_run_checkpoint.bin"
img_size = 448
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载配置
config = CONFIGS[model_type]
config.split = 'overlap'
config.slide_step = 12

# 初始化模型
num_classes = 120
model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)
checkpoint = torch.load(checkpoint_path, map_location=device)
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义类别索引
class_labels = ["Chihuahua", "Japanischer Spaniel", "Malteser", "Pekingese", "Shih-Tzu", "Blenheim-Spaniel",
                "Papillon", "Zwergterrier", "Rhodesian Ridgeback", "Afghanischer Windhund", "Basset", "Beagle",
                "Bluthund", "Bluetick Coonhound", "Schwarzloh Coonhound", "Walker Hound", "Englischer Foxhound",
                "Redbone Coonhound", "Barsoi", "Irischer Wolfshund", "Italienisches Windspiel", "Whippet", "Ibizan Hound",
                "Norwegischer Elchhund", "Otterhund", "Saluki", "Schottischer Hirschhund", "Weimaraner",
                "Staffordshire Bullterrier", "Amerikanischer Staffordshire Terrier", "Bedlington Terrier",
                "Border Terrier", "Kerry Blue Terrier", "Irischer Terrier", "Norfolk Terrier",
                "Norwich Terrier", "Yorkshire Terrier", "Drahthaar-Foxterrier", "Lakeland Terrier",
                "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australischer Terrier", "Dandie Dinmont Terrier",
                "Boston Terrier", "Zwergschnauzer", "Riesenschnauzer", "Standardschnauzer",
                "Schottischer Terrier", "Tibet-Terrier", "Seidenterrier", "Weichhaariger Wheaten Terrier",
                "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Lockenhaariger Retriever",
                "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "Deutsch Kurzhaar",
                "Vizsla", "Englischer Setter", "Irischer Setter", "Gordon Setter", "Bretone", "Clumber Spaniel",
                "Englischer Springer Spaniel", "Walisischer Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                "Irischer Wasserspaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois", "Briard", "Australian Kelpie",
                "Komondor", "Bobtail", "Shetland Sheepdog", "Collie", "Border Collie",
                "Bouvier des Flandres", "Rottweiler", "Deutscher Schäferhund", "Dobermann", "Zwergpinscher",
                "Großer Schweizer Sennenhund", "Berner Sennenhund", "Appenzeller Sennenhund", "Entlebucher Sennenhund",
                "Boxer", "Bullmastiff", "Tibetmastiff", "Französische Bulldogge", "Deutsche Dogge",
                "Bernhardiner", "Eskimohund", "Alaskan Malamute", "Sibirischer Husky", "Affenpinscher", "Basenji",
                "Mops", "Leonberger", "Neufundländer", "Pyrenäenberghund", "Samojede", "Zwergspitz",
                "Chow-Chow", "Keeshond", "Brabanter Griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy-Pudel",
                "Zwergpudel", "Großpudel", "Mexikanischer Nackthund", "Dingo", "Rothund",
                "Afrikanischer Wildhund"]

# 预测函数
def predict(image_path):
    result_label.config(text="Warten...", font=("Helvetica", 14, "bold"))
    root.update_idletasks()  # 强制刷新界面

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        preds = torch.argmax(logits, dim=-1)
    class_name = class_labels[preds.item()]

    result_label.config(text=f"Vorhersage: {class_name}", font=("Helvetica", 14, "bold"))

# GUI 界面
def upload_and_predict():
    file_path = filedialog.askopenfilename(title="Bild auswählen",
                                           filetypes=[("Bilddateien", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        predict(file_path)
    else:
        result_label.config(text="Kein Bild ausgewählt.", font=("Helvetica", 14, "bold"))

# 创建 GUI 界面
root = tk.Tk()
root.title("Hunderassen-Klassifikation")
root.geometry("500x400")
root.resizable(False, False)  # 禁止窗口大小调整

# 设置背景图片
bg_image = Image.open("background.jpg")
bg_image = bg_image.resize((500, 400), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = Label(root, image=bg_photo)
bg_label.place(x=0, y=0, width=500, height=400)

# 上传按钮
upload_button = Button(root, text="Bild hochladen", command=upload_and_predict, font=("Helvetica", 12, "bold"))
upload_button.place(x=180, y=100, width=140, height=40)

# 结果标签
result_label = Label(root, text="Lade ein Hundebild hoch zur Klassifikation", font=("Helvetica", 14, "bold"), bg="lightgray")
result_label.place(x=50, y=200, width=400, height=50)

root.mainloop()
