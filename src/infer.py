# src/infer.py
import argparse
import cv2
import torch
from facenet_pytorch import MTCNN
from src.model import SimpleEmotionCNN
from src.utils import EMO_LABELS
import numpy as np
import torchvision.transforms as T
from PIL import Image

def get_transform(img_size=48):
    return T.Compose([
        T.Resize((img_size,img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def infer_image(model, mtcnn, img_bgr, device, transform, topk=1):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    # detect face boxes and crops
    boxes, probs = mtcnn.detect(pil)
    # mtcnn.detect returns bounding boxes; we will also use mtcnn to extract aligned crop
    crop = mtcnn(pil)  # returns torch tensor for single face if keep_all=False
    outputs = []
    if crop is None:
        return outputs
    # if crop is a tensor with shape (3,H,W)
    if isinstance(crop, torch.Tensor):
        x = crop
    else:
        # If crop is numpy
        x = T.ToTensor()(Image.fromarray(crop))
    x = transform(Image.fromarray((x.permute(1,2,0).numpy()*255).astype('uint8'))).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        top_idx = probs.argmax()
        outputs.append((top_idx, probs[top_idx], probs))
    return outputs, boxes

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device)
    model = SimpleEmotionCNN(num_classes=7).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    transform = get_transform(args.img_size)

    if args.source == 'webcam':
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), "Cannot open webcam"
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results, boxes = infer_image(model, mtcnn, frame, device, transform)
            if results:
                idx, prob, allp = results[0]
                label = EMO_LABELS[idx]
                text = f"{label}: {prob:.2f}"
                if boxes is not None:
                    box = boxes[0].astype(int)
                    cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), (0,255,0), 2)
                    cv2.putText(frame, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Emotion', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        # single image
        img = cv2.imread(args.source)
        if img is None:
            print("Image not found:", args.source); return
        results, boxes = infer_image(model, mtcnn, img, device, transform)
        if not results:
            print("No face detected")
            return
        idx, prob, _ = results[0]
        label = EMO_LABELS[idx]
        print(f"Predicted: {label} ({prob:.2f})")
        if boxes is not None:
            box = boxes[0].astype(int)
            cv2.rectangle(img, (box[0],box[1]),(box[2],box[3]),(0,255,0),2)
            cv2.putText(img, f"{label} {prob:.2f}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.imshow('Result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', default='webcam', help="'webcam' or path to image")
    parser.add_argument('--img-size', type=int, default=48)
    args = parser.parse_args()
    main(args)
