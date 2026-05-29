"""
YOLOv8n 猪+羊双类别检测模型训练 & 导出。

用法:
  1. 先准备数据集: python datasets/livestock/prepare_dataset.py
  2. 训练:          python tools/train_yolov8_livestock.py
  3. 导出 ONNX:     python tools/train_yolov8_livestock.py --export
  4. ATC 转 .om:    在 Atlas 板子上执行 atc 命令（见下方输出）
"""
import argparse
from pathlib import Path


def train(data_yaml: str, epochs: int = 100, imgsz: int = 640, batch: int = 16,
          project: str = "runs/detect", name: str = "livestock"):
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=20,
        device=0,  # GPU; 用 'cpu' 如果没有 GPU
    )
    print(f"\nTraining complete. Best weights: {results.save_dir}/weights/best.pt")
    return str(Path(results.save_dir) / "weights" / "best.pt")


def export_onnx(weights: str, imgsz: int = 640):
    from ultralytics import YOLO

    model = YOLO(weights)
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
    print(f"\nONNX exported: {onnx_path}")

    # 输出 ATC 命令供在 Atlas 板子上执行
    om_name = Path(weights).stem.replace("best", "yolov8n_livestock_fp16")
    print(f"""
=== 在 Atlas 板子上执行以下命令转换 .om ===

scp {onnx_path} HwHiAiUser@192.168.137.100:~/pig_counting/models/

ssh HwHiAiUser@192.168.137.100
cd ~/pig_counting/models
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --model={Path(onnx_path).name} \\
    --framework=5 \\
    --output={om_name} \\
    --soc_version=Ascend310B4 \\
    --precision_mode=force_fp16 \\
    --input_shape="input:1,3,{imgsz},{imgsz}"
""")
    return str(onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/livestock/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--export", action="store_true", help="导出 ONNX（不训练）")
    parser.add_argument("--weights", type=str, default="runs/detect/livestock/weights/best.pt")
    args = parser.parse_args()

    if args.export:
        export_onnx(args.weights, args.imgsz)
    else:
        best = train(args.data, args.epochs, args.imgsz, args.batch)
        print(f"\n训练完成，权重: {best}")
        print(f"导出 ONNX: python tools/train_yolov8_livestock.py --export --weights {best}")
