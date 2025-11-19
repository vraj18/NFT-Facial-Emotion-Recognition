# src/utils.py
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

EMO_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

def save_checkpoint(state, is_best, outdir='checkpoints', fname=None):
    os.makedirs(outdir, exist_ok=True)
    if fname is None:
        fname = f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    path = os.path.join(outdir, fname)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(outdir, 'best.pth')
        torch.save(state, best_path)
    return path

def load_checkpoint(path, model, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optim_state' in ckpt:
        optimizer.load_state_dict(ckpt['optim_state'])
    return ckpt

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=False)
    return acc, report
