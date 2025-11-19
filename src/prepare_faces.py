# src/prepare_faces.py
import os
import argparse
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main(args):
    df = pd.read_csv(args.csv)
    mtcnn = MTCNN(keep_all=False, device='cuda' if args.cuda and torch.cuda.is_available() else 'cpu', selection_method='center-weighted')
    splits = {
        'Training': 'train',
        'PublicTest': 'public_test',
        'PrivateTest': 'private_test'
    }
    for usage, split_name in splits.items():
        df_split = df[df['Usage'] == usage]
        out_split_dir = os.path.join(args.out, split_name)
        ensure_dir(out_split_dir)
        for idx, row in tqdm(df_split.iterrows(), total=len(df_split), desc=split_name):
            label = str(row['emotion'])
            lab_dir = os.path.join(out_split_dir, label)
            ensure_dir(lab_dir)
            pixels = np.fromstring(row['pixels'], dtype=int, sep=' ')
            img = Image.fromarray(pixels.reshape(48,48).astype(np.uint8)).convert('RGB')
            face = mtcnn(img)
            # If face is None, fallback to resized original
            if face is None:
                face_pil = img.resize((args.img_size, args.img_size))
            else:
                # face may be torch tensor or PIL; convert to PIL
                from torchvision.transforms import ToPILImage
                if hasattr(face, 'cpu'):
                    face_pil = ToPILImage()(face.cpu())
                else:
                    face_pil = Image.fromarray(face)
            fname = f"{idx}.png"
            face_pil.resize((args.img_size,args.img_size)).save(os.path.join(lab_dir, fname))

if __name__ == '__main__':
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--img-size', type=int, default=48)
    parser.add_argument('--cuda', action='store_true', help='use cuda for mtcnn')
    args = parser.parse_args()
    main(args)
