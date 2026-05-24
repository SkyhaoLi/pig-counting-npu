"""猪只体重回归训练脚本 — PIGRGB-Weight 数据集 + MobileViT-S

用法（Kaggle / Colab T4）
========================
    pip install timm torch torchvision pillow tqdm
    git clone https://github.com/maweihong/PIGRGB-Weight
    python train_weight_regressor.py \
        --data_root ./PIGRGB-Weight \
        --epochs 30 --batch_size 32

输出
====
    runs/<timestamp>/best.pt           最优 epoch 权重
    runs/<timestamp>/weight_regressor.onnx  推理用 ONNX
    runs/<timestamp>/metrics.json      训练曲线

注意
====
* 本脚本针对 Kaggle T4 单卡设计；多卡需改成 DDP（未集成）。
* 数据集 ImageFolder 结构假定为：
      PIGRGB-Weight/
      ├── RGB_9579/
      │   ├── train/  *.jpg, *_weight.txt
      │   └── val/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import timm
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    print(f"[!] 缺少依赖: {e}. 请先 `pip install timm torch torchvision pillow tqdm`")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class PigWeightDataset(Dataset):
    """PIGRGB-Weight 数据集：图像 + 体重（kg）回归。

    标注约定（与 GitHub 仓库一致）：
        每张图旁有同名 *_weight.txt，第一行写体重（kg）。
    """

    def __init__(self, root: Path, split: str = "train", img_size: int = 224):
        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(f"未找到 split 目录: {self.root}")
        self.images = sorted(self.root.rglob("*.jpg")) + sorted(self.root.rglob("*.png"))
        self.targets = []
        kept = []
        for img_path in self.images:
            w_file = img_path.with_suffix(".txt")
            if w_file.exists():
                try:
                    weight = float(w_file.read_text().strip().splitlines()[0])
                    self.targets.append(weight)
                    kept.append(img_path)
                except (ValueError, IndexError):
                    continue
        self.images = kept
        print(f"[{split}] 有效样本数: {len(self.images)}")

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img), torch.tensor(self.targets[idx], dtype=torch.float32)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def build_model(model_name: str = "mobilevit_s") -> nn.Module:
    """timm 提供 MobileViT-S；将分类头替换为单值回归。"""
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    feat_dim = backbone.num_features
    head = nn.Sequential(
        nn.Linear(feat_dim, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
    )
    return nn.Sequential(backbone, nn.Flatten(), head)


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for img, target in pbar:
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(img).squeeze(-1)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img.size(0)
        n += img.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae = 0.0
    mape_sum = 0.0
    n = 0
    for img, target in tqdm(loader, desc="eval", leave=False):
        img, target = img.to(device), target.to(device)
        pred = model(img).squeeze(-1)
        mae += (pred - target).abs().sum().item()
        mape_sum += ((pred - target).abs() / target.clamp(min=1)).sum().item()
        n += img.size(0)
    return mae / max(n, 1), 100 * mape_sum / max(n, 1)


# ---------------------------------------------------------------------------
# 导出 ONNX
# ---------------------------------------------------------------------------
def export_onnx(model, output_path: Path, img_size: int = 224, device: str = "cpu"):
    model = model.eval().to(device)
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    torch.onnx.export(
        model, dummy, str(output_path),
        opset_version=13,
        input_names=["input"],
        output_names=["weight_kg"],
        dynamic_axes={"input": {0: "batch"}, "weight_kg": {0: "batch"}},
    )
    print(f"[OK] ONNX 已导出: {output_path}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./PIGRGB-Weight/RGB_9579")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--model", type=str, default="mobilevit_s")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Run dir] {run_dir}")
    print(f"[Device] {args.device}")

    train_set = PigWeightDataset(args.data_root, "train", args.img_size)
    val_set = PigWeightDataset(args.data_root, "val", args.img_size)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()

    history = []
    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_mae, val_mape = evaluate(model, val_loader, args.device)
        scheduler.step()
        print(f"[Epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} | val_MAE={val_mae:.2f} kg | val_MAPE={val_mape:.2f}%")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_mae": val_mae, "val_mape": val_mape})

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"  ↳ new best, saved -> {run_dir / 'best.pt'}")

    (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))

    # 加载最优权重并导出 ONNX
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    export_onnx(model, run_dir / "weight_regressor.onnx", img_size=args.img_size)

    print(f"\n[DONE] best_val_MAE = {best_mae:.2f} kg")
    print(f"[NEXT] 在 Atlas 板上执行 ATC 量化:")
    print(f"  atc --model={run_dir}/weight_regressor.onnx \\")
    print(f"      --framework=5 --output=weight_regressor_fp16 \\")
    print(f"      --soc_version=Ascend310B4 --precision_mode=force_fp16 \\")
    print(f"      --input_shape='input:1,3,{args.img_size},{args.img_size}'")


if __name__ == "__main__":
    main()
