# src/eval.py
import argparse
import torch
from torch.utils.data import DataLoader
from src.dataset import FER2013Dataset
from src.model import SimpleEmotionCNN
from src.utils import load_checkpoint, compute_metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleEmotionCNN(num_classes=7).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    ds = FER2013Dataset(args.csv, split='private_test', cached_dir=args.cached_dir, img_size=args.img_size, mtcnn_device='cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())
    acc, report = compute_metrics(y_true, y_pred)
    print("Test accuracy:", acc)
    print("Classification report:\n", report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--csv', default='/Users/vrajborad/Desktop/NFT_PROJECT/data/fer2013.csv')
    parser.add_argument('--cached-dir', default=None)
    parser.add_argument('--img-size', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    main(args)
