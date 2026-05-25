"""扫描 PIGRGB-Weight 数据集，生成 train.csv / val.csv。

数据集真实结构（archive.org / Baidu Pan 下载）：
    PIGRGB-Weight/
      RGB_9579/fold1/<weight>_<id>/<weight>kg_<N>.png    # 9579 张
      RGB_MASK_3394/RGB_3394/<weight>_<id>/<weight>kg_<N>.png   # 3394 张 RGB
      RGB_MASK_3394/MASK_3394/<weight>_<id>/<weight>kg_<N>.png  # 对应 mask

体重从 *文件名* 解析 (例如 73.36kg_1.png → 73.36)。
按 *子目录* 切 train/val，确保同一头猪不会跨 split 泄漏。

用法
====
    python prepare_pigrgb_weight.py \
        --data_root E:/PIGRGB-Weight/full/PIGRGB-Weight \
        --out_dir E:/PIGRGB-Weight/full \
        --val_ratio 0.1
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


WEIGHT_RE = re.compile(r"^([\d.]+)kg_\d+\.(png|jpg|jpeg)$", re.IGNORECASE)


def scan_subset(root: Path, label: str = "") -> dict[str, list[tuple[Path, float]]]:
    """返回 {pig_id: [(img_path, weight_kg), ...]}。

    pig_id 形如 "fold1/73.36_124"（含父目录名），避免不同 fold 同名碰撞。
    """
    groups: dict[str, list[tuple[Path, float]]] = {}
    if not root.exists():
        return groups
    for img in root.rglob("*.png"):
        m = WEIGHT_RE.match(img.name)
        if not m:
            continue
        try:
            w = float(m.group(1))
        except ValueError:
            continue
        # 用相对 root 的父目录路径作 pig_id，确保跨 fold 唯一
        try:
            rel = img.parent.relative_to(root)
            pig_id = f"{label}/{rel}".replace("\\", "/") if label else str(rel).replace("\\", "/")
        except ValueError:
            pig_id = img.parent.name
        groups.setdefault(pig_id, []).append((img, w))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="PIGRGB-Weight 根目录 (包含 RGB_9579/ 与 RGB_MASK_3394/)")
    ap.add_argument("--out_dir", required=True, help="train.csv/val.csv 写出目录")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_mask_rgb", action="store_true",
                    help="也纳入 RGB_MASK_3394/RGB_3394/ 的 RGB 图")
    args = ap.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[Path, str]] = [(root / "RGB_9579", "RGB_9579")]
    if args.include_mask_rgb:
        sources.append((root / "RGB_MASK_3394" / "RGB_3394", "RGB_3394"))

    all_groups: dict[str, list[tuple[Path, float]]] = {}
    for src, label in sources:
        sub = scan_subset(src, label=label)
        for k, v in sub.items():
            all_groups.setdefault(k, []).extend(v)

    if not all_groups:
        raise SystemExit(f"[!] 未在 {[s for s,_ in sources]} 找到任何 *kg_*.png")

    ids = sorted(all_groups.keys())
    random.Random(args.seed).shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_ratio))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])

    def dump(csv_path: Path, id_set: set[str]):
        n_imgs = 0
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "weight_kg", "pig_id"])
            for pid in sorted(id_set):
                for img, weight in all_groups[pid]:
                    w.writerow([str(img.resolve()), f"{weight:.3f}", pid])
                    n_imgs += 1
        return n_imgs

    n_train = dump(out_dir / "train.csv", train_ids)
    n_val_imgs = dump(out_dir / "val.csv", val_ids)

    print(f"[OK] pig IDs total={len(ids)}  train={len(train_ids)}  val={len(val_ids)}")
    print(f"[OK] images train={n_train}  val={n_val_imgs}")
    print(f"[OK] wrote {out_dir/'train.csv'}")
    print(f"[OK] wrote {out_dir/'val.csv'}")


if __name__ == "__main__":
    main()
