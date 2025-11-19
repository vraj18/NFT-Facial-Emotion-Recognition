# src/train.py
import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FER2013Dataset
from model import SimpleEmotionCNN
from utils import save_checkpoint, compute_metrics
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc='Train batches', leave=False):
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(1).detach().cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.detach().cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    acc, _ = compute_metrics(y_true, y_pred)
    return avg_loss, acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc='Eval batches', leave=False):
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(1).detach().cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.detach().cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    acc, _ = compute_metrics(y_true, y_pred)
    return avg_loss, acc

def make_loaders(csv_path, cached_dir, img_size, batch_size, num_workers, mtcnn_device):
    train_ds = FER2013Dataset(csv_path, split='train', cached_dir=cached_dir, img_size=img_size, mtcnn_device=mtcnn_device)
    val_ds = FER2013Dataset(csv_path, split='public_test', cached_dir=cached_dir, img_size=img_size, mtcnn_device=mtcnn_device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    mtcnn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader = make_loaders(args.data_csv, args.cached_dir, args.img_size, args.batch_size, args.workers, mtcnn_device)
    model = SimpleEmotionCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s")
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'val_acc': val_acc
        }
        save_checkpoint(ckpt, is_best, outdir=args.ckpt_dir, fname=f"epoch_{epoch}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', default='/Users/vrajborad/Desktop/Code/NFT_PROJECT/data/fer2013.csv')
    parser.add_argument('--cached-dir', default=None, help='path to cached faces dir (optional)')
    parser.add_argument('--img-size', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ckpt-dir', default='checkpoints')
    args = parser.parse_args()
    main(args)
