"""
下载并合并猪+羊数据集，生成 YOLO 格式训练数据。

数据源:
  1. Roboflow Universe 上的 pig detection / sheep detection 数据集
  2. 或手动下载后放到 datasets/livestock/raw/ 下

用法:
  python datasets/livestock/prepare_dataset.py --source roboflow
  python datasets/livestock/prepare_dataset.py --source local --pig_dir raw/pig --sheep_dir raw/sheep
"""
import argparse
import shutil
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent
IMG_TRAIN = BASE_DIR / "images" / "train"
IMG_VAL = BASE_DIR / "images" / "val"
LBL_TRAIN = BASE_DIR / "labels" / "train"
LBL_VAL = BASE_DIR / "labels" / "val"

VAL_RATIO = 0.2


def ensure_dirs():
    for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
        d.mkdir(parents=True, exist_ok=True)


def remap_yolo_labels(label_dir: Path, new_class_id: int, out_lbl: Path, out_img: Path,
                      img_dir: Path, img_ext: str = ".jpg"):
    """将 YOLO 标签中的 class_id 替换为 new_class_id，复制图片和标签。"""
    count = 0
    for lbl_file in label_dir.glob("*.txt"):
        img_file = img_dir / (lbl_file.stem + img_ext)
        if not img_file.exists():
            # 尝试其他扩展名
            for ext in [".png", ".jpeg", ".bmp"]:
                img_file = img_dir / (lbl_file.stem + ext)
                if img_file.exists():
                    break
            else:
                continue

        # 重映射标签
        new_lines = []
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = str(new_class_id)
                    new_lines.append(" ".join(parts))

        if not new_lines:
            continue

        # 写入新标签
        with open(out_lbl / lbl_file.name, "w") as f:
            f.write("\n".join(new_lines) + "\n")

        # 复制图片
        shutil.copy2(img_file, out_img / img_file.name)
        count += 1

    return count


def split_train_val(img_dir: Path, lbl_dir: Path):
    """将 train 目录中的数据按比例分出 val 集。"""
    imgs = list(img_dir.glob("*.*"))
    random.seed(42)
    random.shuffle(imgs)
    val_count = int(len(imgs) * VAL_RATIO)

    for img in imgs[:val_count]:
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            shutil.move(str(img), str(IMG_VAL / img.name))
            shutil.move(str(lbl), str(LBL_VAL / lbl.name))


def prepare_from_local(pig_dir: str, sheep_dir: str, pig_ext: str = ".jpg", sheep_ext: str = ".jpg"):
    """从本地目录合并猪和羊数据集。"""
    ensure_dirs()
    pig_path = Path(pig_dir)
    sheep_path = Path(sheep_dir)

    # 猪: class_id=0
    pig_count = 0
    if pig_path.exists():
        lbl_dir = pig_path / "labels" if (pig_path / "labels").exists() else pig_path
        img_dir = pig_path / "images" if (pig_path / "images").exists() else pig_path
        pig_count = remap_yolo_labels(lbl_dir, 0, LBL_TRAIN, IMG_TRAIN, img_dir, pig_ext)
        print(f"[pig] {pig_count} images mapped to class 0")

    # 羊: class_id=1
    sheep_count = 0
    if sheep_path.exists():
        lbl_dir = sheep_path / "labels" if (sheep_path / "labels").exists() else sheep_path
        img_dir = sheep_path / "images" if (sheep_path / "images").exists() else sheep_path
        sheep_count = remap_yolo_labels(lbl_dir, 1, LBL_TRAIN, IMG_TRAIN, img_dir, sheep_ext)
        print(f"[sheep] {sheep_count} images mapped to class 1")

    # 划分 train/val
    split_train_val(IMG_TRAIN, LBL_TRAIN)

    train_imgs = len(list(IMG_TRAIN.glob("*.*")))
    val_imgs = len(list(IMG_VAL.glob("*.*")))
    print(f"\nDataset ready: {train_imgs} train, {val_imgs} val")
    print(f"Config: {BASE_DIR / 'data.yaml'}")


def prepare_from_roboflow():
    """提示用户从 Roboflow 下载数据集。"""
    print("""
=== Roboflow 数据集下载指南 ===

1. 猪检测数据集:
   访问 https://universe.roboflow.com/search?q=pig+detection&task=objectDetection
   选择一个项目 → Download → YOLOv8 格式 → 解压到 datasets/livestock/raw/pig/

2. 羊检测数据集:
   访问 https://universe.roboflow.com/search?q=sheep+detection&task=objectDetection
   选择一个项目 → Download → YOLOv8 格式 → 解压到 datasets/livestock/raw/sheep/

3. 然后运行:
   python datasets/livestock/prepare_dataset.py --source local --pig_dir datasets/livestock/raw/pig --sheep_dir datasets/livestock/raw/sheep

或者使用 Kaggle Animals Detection Dataset:
   https://www.kaggle.com/datasets/antoreepjana/animals-detection-data
   筛选 pig 和 sheep 类别，转换为 YOLO 格式。
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["roboflow", "local"], default="roboflow")
    parser.add_argument("--pig_dir", type=str, default="")
    parser.add_argument("--sheep_dir", type=str, default="")
    parser.add_argument("--pig_ext", type=str, default=".jpg")
    parser.add_argument("--sheep_ext", type=str, default=".jpg")
    args = parser.parse_args()

    if args.source == "roboflow":
        prepare_from_roboflow()
    else:
        if not args.pig_dir or not args.sheep_dir:
            print("Error: --pig_dir and --sheep_dir required for local source")
            exit(1)
        prepare_from_local(args.pig_dir, args.sheep_dir, args.pig_ext, args.sheep_ext)
