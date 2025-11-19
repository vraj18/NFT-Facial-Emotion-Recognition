# src/dataset.py
import os
import io
import math
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm
import torch

EMO_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

class FER2013Dataset(Dataset):
    """
    Dataset that reads fer2013 csv and returns cropped face tensors.
    If cached_dir provided, loads pre-cropped face images from disk organized as:
      cached_dir/{split}/{label}/*.png
    Otherwise it uses simple center cropping (MTCNN disabled due to errors).
    """
    def __init__(self, csv_path=None, split='train', cached_dir=None, img_size=48, transform=None, mtcnn_device='cpu'):
        assert split in ('train', 'public_test', 'private_test')
        self.split = split
        self.cached_dir = cached_dir
        self.img_size = img_size
        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        if cached_dir:
            # Build file list from cached_dir
            self.samples = []
            base = os.path.join(cached_dir, split)
            if not os.path.isdir(base):
                raise FileNotFoundError(f"Cached split dir not found: {base}")
            for label in sorted(os.listdir(base)):
                lab_dir = os.path.join(base, label)
                if not os.path.isdir(lab_dir): continue
                for fname in os.listdir(lab_dir):
                    if fname.lower().endswith(('.png','.jpg','.jpeg')):
                        self.samples.append((os.path.join(lab_dir,fname), int(label)))
        else:
            # Read CSV and keep only rows for given split
            if not csv_path:
                raise ValueError("csv_path required if cached_dir not provided")
            df = pd.read_csv(csv_path)
            df_split = df[df['Usage'] == self._usage_from_split(split)]
            self.df = df_split.reset_index(drop=True)
            self.samples = list(range(len(self.df)))
            
            # MTCNN is disabled due to persistent errors
            # We'll use simple center cropping instead
            print("MTCNN disabled - using simple image processing")

    def _usage_from_split(self, split):
        return {
            'train': 'Training',
            'public_test': 'PublicTest',
            'private_test': 'PrivateTest'
        }[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cached_dir:
            path, label = self.samples[idx]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img, label
        else:
            row = self.df.iloc[idx]
            label = int(row['emotion'])
            pixels = np.fromstring(row['pixels'], dtype=int, sep=' ')
            img = Image.fromarray(pixels.reshape(48,48).astype(np.uint8))
            
            # Convert to RGB and apply simple processing
            img = img.convert('RGB')
            
            # Simple center crop and resize (instead of MTCNN)
            # This is much more reliable and avoids the MTCNN errors
            width, height = img.size
            
            # Simple center crop to focus on face area
            crop_size = min(width, height)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            
            img_cropped = img.crop((left, top, right, bottom))
            img_resized = img_cropped.resize((self.img_size, self.img_size))
            
            img_t = self.transform(img_resized)
            return img_t, label