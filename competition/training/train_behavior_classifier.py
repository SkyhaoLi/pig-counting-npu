"""猪只行为分类训练脚本 — TSN (Temporal Segment Network) + ResNet50

参考论文
========
Yang et al. (Sensors 2020) Two-Stream Convolutional Networks for Pig Behavior
论文目标精度：5 类行为 Top-1 ≈ 98.99% on China-Agri-Uni-1000.

行为类别（默认）
================
    feeding, lying, walking, scratching, mounting

数据集结构
==========
    behavior_dataset/
    ├── train/
    │   ├── feeding/   *.mp4
    │   ├── lying/     *.mp4
    │   └── ...
    └── val/

用法
====
    pip install torch torchvision av einops tqdm
    python train_behavior_classifier.py --data_root ./behavior_dataset
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models import resnet50, ResNet50_Weights
    import av
    from tqdm import tqdm
except ImportError as e:
    print(f"[!] 缺少依赖: {e}.  pip install torch torchvision av einops tqdm")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# TSN-style 视频数据集（采样 N 段，每段 1 帧）
# ---------------------------------------------------------------------------
class TSNVideoDataset(Dataset):
    def __init__(self, root: Path, classes: list, num_segments: int = 8,
                 img_size: int = 224, split: str = "train"):
        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.classes = classes
        self.num_segments = num_segments

        self.samples = []
        for cls_idx, cls_name in enumerate(classes):
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                print(f"[!] 缺少类别目录: {cls_dir}")
                continue
            for vid in cls_dir.glob("*.mp4"):
                self.samples.append((str(vid), cls_idx))
        print(f"[{split}] 视频数: {len(self.samples)}")

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if split == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, path: str) -> list:
        """TSN：将视频均分 N 段，每段取中间一帧。"""
        container = av.open(path)
        stream = container.streams.video[0]
        total = stream.frames or 0
        if total == 0:
            # fallback: 解码计数
            total = sum(1 for _ in container.decode(stream))
            container.close()
            container = av.open(path)
            stream = container.streams.video[0]

        if total < self.num_segments:
            indices = list(range(total)) + [total - 1] * (self.num_segments - total)
        else:
            seg = total // self.num_segments
            indices = [seg // 2 + i * seg for i in range(self.num_segments)]

        target_set = set(indices)
        frames = {}
        for i, frame in enumerate(container.decode(stream)):
            if i in target_set:
                frames[i] = frame.to_ndarray(format="rgb24")
            if len(frames) == len(target_set):
                break
        container.close()
        return [frames[i] for i in indices]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            arrs = self._sample_frames(path)
        except Exception as e:
            print(f"[!] 解码失败 {path}: {e}")
            arrs = [torch.zeros(3, 224, 224).numpy().transpose(1, 2, 0)] * self.num_segments
        tensors = [self.transform(arr) for arr in arrs]
        clip = torch.stack(tensors, dim=0)  # (T, C, H, W)
        return clip, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# 模型：ResNet50 + 时间段平均融合（TSN consensus）
# ---------------------------------------------------------------------------
class TSNResNet50(nn.Module):
    def __init__(self, num_classes: int, num_segments: int = 8):
        super().__init__()
        self.num_segments = num_segments
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, clip):  # clip: (B, T, C, H, W)
        b, t = clip.shape[:2]
        flat = clip.view(b * t, *clip.shape[2:])
        feat = self.backbone(flat)
        feat = feat.view(b, t, -1).mean(dim=1)  # consensus
        return self.classifier(feat)


# ---------------------------------------------------------------------------
# 训练 / 评估循环
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, desc="train", leave=False)
    for clip, label in pbar:
        clip, label = clip.to(device), label.to(device)
        opt.zero_grad()
        logits = model(clip)
        loss = crit(logits, label)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * clip.size(0)
        correct += (logits.argmax(1) == label).sum().item()
        total += clip.size(0)
    return loss_sum / max(total, 1), 100 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for clip, label in tqdm(loader, desc="eval", leave=False):
        clip, label = clip.to(device), label.to(device)
        logits = model(clip)
        correct += (logits.argmax(1) == label).sum().item()
        total += clip.size(0)
    return 100 * correct / max(total, 1)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--classes", type=str,
                        default="feeding,lying,walking,scratching,mounting")
    parser.add_argument("--num_segments", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    classes = args.classes.split(",")
    run_dir = Path("runs") / f"behavior_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run dir] {run_dir} | classes={classes}")

    train_set = TSNVideoDataset(args.data_root, classes, args.num_segments,
                                args.img_size, "train")
    val_set = TSNVideoDataset(args.data_root, classes, args.num_segments,
                              args.img_size, "val")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = TSNResNet50(len(classes), args.num_segments).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    history = []
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, crit, args.device)
        val_acc = evaluate(model, val_loader, args.device)
        sched.step()
        print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% "
              f"| val_acc={val_acc:.2f}%")
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"  ↳ new best, saved.")

    (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))
    (run_dir / "classes.txt").write_text("\n".join(classes))

    print(f"\n[DONE] best_val_acc = {best_acc:.2f}%")


if __name__ == "__main__":
    main()
