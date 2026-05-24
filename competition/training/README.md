# 训练脚本 — Kaggle / Colab T4 就绪

本目录提供三个训练任务的端到端脚本，配合本项目的健康预警模块使用。

## 任务总览

| 脚本 | 目标模型 | 数据集 | 预估训练时长 (T4) | 输出 |
|------|---------|--------|----------------|------|
| `train_weight_regressor.py` | MobileViT-S 体重回归 | PIGRGB-Weight (9579 张) | 2–3 h | `weight_regressor.onnx` |
| `train_behavior_classifier.py` | TSN-ResNet50 行为分类 | China-Agri-Uni-1000 (自托管/合作申请) | 4–5 h | `behavior_classifier.onnx` |
| `finetune_vjepa.py` | V-JEPA 2 LoRA 微调 | 自有未标注视频 + 少量标注 | 6–8 h | `vjepa_pig_lora.pt` |

## 一键运行（Kaggle 示例）

```python
# Kaggle Notebook：选择 T4 加速器，复制以下单元到 notebook
!git clone https://github.com/<your-account>/pig_couter.git
%cd pig_couter/competition/training
!pip install -q ultralytics timm torch torchvision tqdm
!python train_weight_regressor.py --epochs 30 --batch_size 32
```

## Colab 示例

```bash
!nvidia-smi
%cd /content
!git clone https://github.com/<your-account>/pig_couter.git
%cd pig_couter/competition/training
!python train_weight_regressor.py --epochs 30 --batch_size 32 --device cuda
```

## 在板端 NPU 部署训练后模型

PyTorch / ONNX → Atlas Ascend OM 转换：

```bash
# 1. 训练完成后会得到 weight_regressor.onnx
# 2. 在 Atlas 板上 source CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. ATC 量化
atc --model=weight_regressor.onnx \
    --framework=5 \
    --output=weight_regressor_fp16 \
    --soc_version=Ascend310B4 \
    --precision_mode=force_fp16 \
    --input_shape="input:1,3,224,224"

# 4. 注入到健康预警模块
python -c "
from health_module import set_weight_model
from onnx_runner import OnnxRunner
model = OnnxRunner('weight_regressor_fp16.om')
set_weight_model(model)
"
```

## 数据集获取

* **PIGRGB-Weight**：`git clone https://github.com/maweihong/PIGRGB-Weight` — 9579 张 RGB + 标注
* **CV4PigBW**：`git clone https://github.com/yebigithub/CV4PigBW` — 工业级视频体重数据
* **Edinburgh Pig Behavior**：参考 IEEE 2025 Animal-JEPA 论文附录获取
* **China-Agri-Uni-1000**：联系中国农业大学计算机系协作申请

## 注意

* 本目录脚本未在本地执行（无 GPU），但所有依赖、超参、数据加载均按 Kaggle T4 标准环境校准。
* 训练日志会写入 `runs/<timestamp>/`，并支持断点续训。
* 训练完成后 ONNX → ATC 转 OM 的指令模板见 `export_onnx.py`。
