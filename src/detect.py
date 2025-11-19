# ------------------------------
# Advanced Fuzzy Emotion System
# ------------------------------
import numpy as np

class AdvancedFuzzyEmotion:
    def __init__(self):
        self.rules = self._setup_rules()
        self.emotion_mixtures = self._setup_emotion_mixtures()
    
    def _setup_emotion_mixtures(self):
        mixtures = {
            'Happy+Surprise': {'name': 'Excited', 'conditions': lambda p: p['Happy']>0.4 and p['Surprise']>0.3, 'confidence_threshold':0.6},
            'Fear+Surprise': {'name': 'Anxious', 'conditions': lambda p: p['Fear']>0.3 and p['Surprise']>0.3, 'confidence_threshold':0.5},
            'Sad+Angry': {'name': 'Frustrated', 'conditions': lambda p: p['Sad']>0.3 and p['Angry']>0.3, 'confidence_threshold':0.5},
            'Happy+Neutral': {'name': 'Content', 'conditions': lambda p: p['Happy']>0.4 and p['Neutral']>0.4, 'confidence_threshold':0.6},
            'Sad+Neutral': {'name': 'Melancholic', 'conditions': lambda p: p['Sad']>0.4 and p['Neutral']>0.4, 'confidence_threshold':0.6},
            'Angry+Disgust': {'name': 'Contempt', 'conditions': lambda p: p['Angry']>0.4 and p['Disgust']>0.2, 'confidence_threshold':0.5},
            'Fear+Sad': {'name': 'Worried', 'conditions': lambda p: p['Fear']>0.3 and p['Sad']>0.3, 'confidence_threshold':0.5}
        }
        return mixtures
    
    def _setup_rules(self):
        rules = [
            {'name':'Clear Happy', 'conditions': lambda p: p['Happy']>0.7 and p['Neutral']<0.2, 'action':'Happy', 'confidence':'VERY_HIGH'},
            {'name':'Clear Sad', 'conditions': lambda p: p['Sad']>0.6 and p['Happy']<0.1, 'action':'Sad', 'confidence':'HIGH'},
            {'name':'Clear Angry', 'conditions': lambda p: p['Angry']>0.6 and p['Fear']<0.2, 'action':'Angry', 'confidence':'HIGH'},
            {'name':'Clear Neutral', 'conditions': lambda p: p['Neutral']>0.7 and max(p['Happy'],p['Sad'],p['Angry'])<0.2, 'action':'Neutral', 'confidence':'HIGH'},
            {'name':'Happy Surprise', 'conditions': lambda p: p['Happy']>0.4 and p['Surprise']>0.3, 'action':'Excited', 'confidence':'MEDIUM'},
            {'name':'Fear Surprise', 'conditions': lambda p: p['Fear']>0.3 and p['Surprise']>0.3, 'action':'Anxious', 'confidence':'MEDIUM'},
            {'name':'Weighted Decision', 'conditions': lambda p: True, 'action': lambda p: max(p,key=p.get), 'confidence':'LOW'}
        ]
        return rules
    
    def analyze_emotion(self, probabilities):
        # Check mixtures first
        for key, mix in self.emotion_mixtures.items():
            if mix['conditions'](probabilities):
                confidence = self._calculate_mixture_confidence(probabilities, key)
                if confidence >= mix['confidence_threshold']:
                    return mix['name'], confidence, f"Mixture {key}"
        
        # Apply standard rules
        for rule in self.rules:
            if rule['conditions'](probabilities):
                action = rule['action'](probabilities) if callable(rule['action']) else rule['action']
                confidence = self._calculate_confidence(probabilities, action)
                return action, confidence, rule['name']
        
        # Default
        action = max(probabilities, key=probabilities.get)
        confidence = self._calculate_confidence(probabilities, action)
        return action, confidence, "Default"
    
    def _calculate_confidence(self, probabilities, emotion):
        target_prob = probabilities.get(emotion, 0)
        other_probs = [p for e,p in probabilities.items() if e!=emotion]
        dominance = target_prob - max(other_probs) if other_probs else target_prob
        conf = (target_prob*0.7 + dominance*0.3)
        return max(0.1, min(0.99, conf))
    
    def _calculate_mixture_confidence(self, probabilities, mixture_key):
        emotions = mixture_key.split('+')
        avg_prob = np.mean([probabilities[e] for e in emotions])
        other_probs = [p for e,p in probabilities.items() if e not in emotions]
        max_other = max(other_probs) if other_probs else 0
        dominance_ratio = avg_prob / max_other if max_other>0 else 2.0
        conf = avg_prob*0.6 + min(1.0, dominance_ratio*0.5)*0.4
        return max(0.1, min(0.99, conf))

# Initialize
fuzzy_system = AdvancedFuzzyEmotion()


# detect.py
import os
import threading
import time
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from facenet_pytorch import MTCNN
import pyttsx3
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

# ------------------------------
# Device setup
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("🖥️ Device:", device)

# ------------------------------
# Load your trained model
# ------------------------------
from model import SimpleEmotionCNN

model = SimpleEmotionCNN(num_classes=7).to(device)
checkpoint_path = "/Users/vrajborad/Desktop/Code/NFT_PROJECT/src/checkpoints/best.pth"  # <- your trained weights
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"✅ Loaded model from {checkpoint_path}")
else:
    raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ------------------------------
# MTCNN for face detection
# ------------------------------
mtcnn = MTCNN(keep_all=True, device=device, image_size=160)

# ------------------------------
# Fuzzy Logic class (paste your AdvancedFuzzyEmotion here)
# ------------------------------
# Paste the full AdvancedFuzzyEmotion class you shared earlier here
# ------------------------------
# For brevity in this snippet, we assume fuzzy_system is already defined
from detect_fuzzy_class import AdvancedFuzzyEmotion  # remove if you paste class above

fuzzy_system = AdvancedFuzzyEmotion()

# ------------------------------
# Text-to-Speech using pyttsx3
# ------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

# ------------------------------
# GUI setup
# ------------------------------
COLORS = {
    "bg": "#f5f5f5",
    "card_bg": "#ffffff",
    "accent": "#2c3e50",
    "button_bg": "#000000",
    "button_hover": "#333333",
    "success": "#27ae60",
    "warning": "#e74c3c",
    "text_primary": "#2c3e50",
    "text_secondary": "#7f8c8d",
    "border": "#bdc3c7"
}

root = tk.Tk()
root.title("Emotion Detection with Fuzzy Logic")
root.geometry("900x750")
root.configure(bg=COLORS["bg"])

main_frame = tk.Frame(root, bg=COLORS["bg"])
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

title_label = tk.Label(main_frame, text="🎭 Advanced Emotion Detection", 
                       font=("Arial", 20, "bold"), fg=COLORS["accent"], bg=COLORS["bg"])
title_label.pack(pady=10)

# Webcam display
video_label = tk.Label(main_frame, bg="black", width=640, height=360)
video_label.pack(pady=10)

# Emotion display
emotion_label = tk.Label(main_frame, text="Press START", font=("Arial", 18, "bold"), 
                         fg=COLORS["success"], bg=COLORS["card_bg"])
emotion_label.pack(pady=10)

stats_label = tk.Label(main_frame, text="Fuzzy Confidence: -- | Detections: 0 | Last Rule: --",
                       font=("Arial", 12), fg=COLORS["text_secondary"], bg=COLORS["card_bg"])
stats_label.pack(pady=5)

progress = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="determinate")
progress.pack(pady=10)

# ------------------------------
# Global variables
# ------------------------------
cap = None
running = False
detection_count = 0
last_detection_time = 0
last_emotion = "None"
last_rule = "None"

# ------------------------------
# Core functions
# ------------------------------
def open_webcam():
    global cap
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Webcam {i} opened")
            return True
    return False

def draw_face_box(frame, boxes):
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(c) for c in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def preprocess_face(face_tensor):
    face_tensor = face_tensor.unsqueeze(0).to(device)
    face_tensor = F.interpolate(face_tensor, size=(48,48), mode='bilinear', align_corners=False)
    face_tensor = face_tensor / 255.0
    return face_tensor

def process_face_emotion(face_tensor):
    global detection_count, last_emotion, last_rule

    face_tensor = preprocess_face(face_tensor)
    with torch.no_grad():
        output = model(face_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    cnn_probs = {emo: float(probs[i]) for i, emo in enumerate(EMOTIONS)}
    refined_emotion, confidence, rule = fuzzy_system.analyze_emotion(cnn_probs)
    
    detection_count += 1
    last_emotion = refined_emotion
    last_rule = rule
    
    # Update UI
    color = COLORS["success"] if confidence > 0.6 else COLORS["warning"]
    emotion_label.config(text=f"🧠 Emotion: {refined_emotion}", fg=color)
    stats_label.config(text=f"Fuzzy Confidence: {confidence:.2f} | Detections: {detection_count} | Rule: {rule[:25]}...")
    progress['value'] = (progress['value'] + 15) % 100
    if detection_count % 3 == 1:
        speak(f"Detected emotion is {refined_emotion}")

def update_frame():
    global last_detection_time
    if not running or cap is None:
        return
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        boxes, _ = mtcnn.detect(img)
        frame = draw_face_box(frame, boxes)
        
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640,360)))
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        
        # Process every 2 sec
        if time.time() - last_detection_time > 2 and boxes is not None:
            last_detection_time = time.time()
            face_tensors = mtcnn(img)
            if face_tensors is not None:
                if face_tensors.dim() == 3:
                    face_tensors = face_tensors.unsqueeze(0)
                for face_tensor in face_tensors:
                    threading.Thread(target=lambda ft=face_tensor: process_face_emotion(ft), daemon=True).start()
    root.after(20, update_frame)

def start_detection():
    global running
    if running:
        return
    if not open_webcam():
        messagebox.showerror("Error", "Could not open webcam")
        return
    running = True
    update_frame()

def stop_detection():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None

# Buttons
start_btn = tk.Button(main_frame, text="Start Detection", command=start_detection)
start_btn.pack(pady=5)
stop_btn = tk.Button(main_frame, text="Stop Detection", command=stop_detection)
stop_btn.pack(pady=5)

root.mainloop()
