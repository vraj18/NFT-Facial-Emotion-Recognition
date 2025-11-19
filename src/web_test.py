# src/webcam_detect.py
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from model import SimpleEmotionCNN
from utils import EMO_LABELS
from facenet_pytorch import MTCNN
print("Facenet-pytorch imported successfully!")
# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "/Users/vrajborad/Desktop/Code/NFT_PROJECT/src/checkpoints/best.pth"
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
# FACE DETECTOR
# ---------------------------
mtcnn = MTCNN(keep_all=True, device=device)

# ---------------------------
# PREPROCESSING
# ---------------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # [0, 1]
])

# ---------------------------
# PREDICT FUNCTION
# ---------------------------
@torch.no_grad()
def predict_emotion(face_rgb):
    """
    Takes RGB cropped face (numpy array)
    Returns predicted emotion label
    """
    img = preprocess(face_rgb).unsqueeze(0).to(device)
    logits = model(img)
    pred = torch.argmax(logits, dim=1).item()
    return EMO_LABELS[pred]

# ---------------------------
# WEBCAM LOOP
# ---------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Crop face safely
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size == 0:
                continue

            # Convert to RGB for model input
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Predict emotion
            emotion = predict_emotion(face_rgb)

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Emotion Detection - Webcam", frame)

    # Quit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
