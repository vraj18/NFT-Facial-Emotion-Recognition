# src/detect.py
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from model import SimpleEmotionCNN
from utils import EMO_LABELS, load_checkpoint

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "/Users/vrajborad/Desktop/Code/NFT_PROJECT/src/checkpoints/best.pth"
IMAGE_PATH = "happy.jpg"    # change to your test image
IMG_SIZE = 48

# ---------------------------
# LOAD MODEL
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SimpleEmotionCNN(num_classes=7)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

print("Model loaded successfully!")

# ---------------------------
# PREPROCESSING PIPELINE
# matches FER2013 training
# ---------------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),   # [0,1]
])

# ---------------------------
# EMOTION PREDICTION FUNCTION
# ---------------------------
@torch.no_grad()
def predict_emotion(image_bgr):
    """
    Takes a BGR image (OpenCV)
    Returns predicted label string
    """
    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess
    tensor = preprocess(img_rgb).unsqueeze(0).to(device)

    # Forward pass
    logits = model(tensor)
    pred = torch.argmax(logits, dim=1).item()

    return EMO_LABELS[pred]

# ---------------------------
# LOAD IMAGE
# ---------------------------
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# ---------------------------
# PREDICT
# ---------------------------
emotion = predict_emotion(img)
print("Predicted Emotion:", emotion)

# Show image for verification
cv2.imshow("Input Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
