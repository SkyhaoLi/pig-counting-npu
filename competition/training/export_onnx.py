"""ONNX/OM 导出与 Atlas 部署工具脚本。

功能
====
1. PyTorch .pt → ONNX
2. ONNX → Atlas OM（命令模板生成，需要在 Atlas 板上执行）
3. 模型推理速度基准（CPU / CUDA / NPU）
"""
from __future__ import annotations

import argparse
from pathlib import Path

ATC_TEMPLATE = """
# === Atlas 板端执行（OM 量化）===
# 1) 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2) ATC 量化转换
atc --model={onnx_path} \\
    --framework=5 \\
    --output={om_name} \\
    --soc_version=Ascend310B4 \\
    --precision_mode=force_fp16 \\
    --input_shape="input:1,3,{img_size},{img_size}" \\
    --log=info

# 3) 验证
ls -lh {om_name}.om

# 4) Python 推理
python -c "
from npu_detector import NPUDetector  # 复用项目封装
runner = NPUDetector('{om_name}.om')
print(runner)
"
"""


def export(pt_path: Path, output_path: Path, img_size: int = 224):
    import torch
    from train_weight_regressor import build_model
    model = build_model()
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(model, dummy, str(output_path),
                      opset_version=13,
                      input_names=["input"], output_names=["weight_kg"],
                      dynamic_axes={"input": {0: "batch"}})
    print(f"[OK] ONNX: {output_path}")
    print(ATC_TEMPLATE.format(
        onnx_path=output_path,
        om_name=output_path.stem + "_fp16",
        img_size=img_size,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, required=True)
    parser.add_argument("--out", type=str, default="weight_regressor.onnx")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    export(Path(args.pt), Path(args.out), args.img_size)


if __name__ == "__main__":
    main()
