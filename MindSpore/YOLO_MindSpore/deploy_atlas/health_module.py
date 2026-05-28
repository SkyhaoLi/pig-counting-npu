"""猪只健康预警与体重估计模块（启发式 + 可替换为训练模型）。

设计目标
========
* 输入：ByteTrack 输出的每条轨迹的 box 尺寸序列、位置序列、置信度序列、帧率
* 输出：体重(kg)、姿态、活动度评分、健康综合评分、异常标记
* 接口：纯函数 + 可替换 ModelInterface（占位），便于训练完成后无缝切换

启发式公式（参赛阶段使用）
============================
* 体重估计：median(框面积) / weight_scale  → 体重(kg)，scale 默认 800.0
  - 参考 CV4PigBW (Bi et al. 2025) 经验：体长×体宽与体重高相关
* 姿态判别：median(宽高比) → standing / lying_side / lying_belly
* 活动度：总位移 / 时长 → px/s，归一到 [0,1]
* 健康评分：0.6·活动度 + 0.3·姿态权重 - 0.1·异常姿态比例

训练模型接入
============
当 `competition/training/` 训练完成 MobileViT-S 体重模型与 V-JEPA 微调权重后，
可通过 `set_weight_model(model)` 与 `set_health_model(model)` 注入。
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 可替换模型接口（预留）
# ----------------------------------------------------------------------
_WEIGHT_MODEL = None  # 训练好的体重回归模型（MobileViT-S onnx）
_HEALTH_MODEL = None  # 训练好的行为分类 / V-JEPA 微调权重


def set_weight_model(model) -> None:
    """注入训练好的体重模型（须实现 predict(roi_image) -> float）。"""
    global _WEIGHT_MODEL
    _WEIGHT_MODEL = model


def set_health_model(model) -> None:
    """注入训练好的健康/行为模型。"""
    global _HEALTH_MODEL
    _HEALTH_MODEL = model


# ----------------------------------------------------------------------
# 体重估计
# ----------------------------------------------------------------------
def estimate_weight_kg(box_sizes: Sequence[Tuple[float, float, int]],
                       weight_scale: float = 800.0,
                       frame_area: Optional[float] = None) -> float:
    """根据 ByteTrack 框面积估计体重（kg）。

    Args:
        box_sizes: [(w, h, frame_idx), ...] 序列
        weight_scale: 像素面积到 kg 的折算系数（经验默认 800.0）
        frame_area: 如果给出，会做透视校正（remote pig has smaller box）

    Returns:
        体重 kg；若无框则返回 0.0
    """
    if not box_sizes:
        return 0.0

    areas = np.array([w * h for w, h, _ in box_sizes], dtype=np.float32)
    median_area = float(np.median(areas))

    if frame_area and frame_area > 0:
        # 简单透视校正：以画面面积的 5% 为参考尺度
        ref = frame_area * 0.05
        median_area = median_area * (ref / max(median_area, 1.0)) ** 0.2

    weight_kg = median_area / max(weight_scale, 1e-3)
    # 合理范围：5kg 仔猪 - 350kg 种猪
    return float(np.clip(weight_kg, 5.0, 350.0))


# ----------------------------------------------------------------------
# 姿态识别
# ----------------------------------------------------------------------
def compute_posture(box_sizes: Sequence[Tuple[float, float, int]]) -> str:
    """根据框宽高比判别姿态。

    标准：宽/高 > 1.5 → 侧卧；< 0.7 → 站立；其余 → 俯卧 / 蹲
    """
    if not box_sizes:
        return "unknown"
    ratios = np.array([w / max(h, 1.0) for w, h, _ in box_sizes], dtype=np.float32)
    median_ratio = float(np.median(ratios))
    if median_ratio > 1.5:
        return "lying_side"
    if median_ratio < 0.7:
        return "standing"
    return "lying_belly"


def posture_entropy(box_sizes: Sequence[Tuple[float, float, int]]) -> float:
    """姿态多样性熵：健康猪通常在多姿态间切换，发病猪倾向单一姿态。"""
    if len(box_sizes) < 3:
        return 0.0
    ratios = np.array([w / max(h, 1.0) for w, h, _ in box_sizes])
    bins = np.array([0.7, 1.5])
    hist, _ = np.histogram(ratios, bins=[0, 0.7, 1.5, 1e9])
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if len(p) else 0.0


# ----------------------------------------------------------------------
# 活动度
# ----------------------------------------------------------------------
def compute_activity(positions: Sequence[Tuple[float, float, int]],
                     fps: float,
                     speed_norm: float = 50.0) -> float:
    """活动度评分：0(静止) ~ 1(活跃)。

    speed_norm 默认 50 px/s ≈ 健康猪正常游走速度。
    """
    if len(positions) < 2 or fps <= 0:
        return 0.0
    pts = np.array(positions, dtype=np.float32)
    diffs = np.diff(pts[:, :2], axis=0)
    dist = float(np.sum(np.linalg.norm(diffs, axis=1)))
    duration = (pts[-1, 2] - pts[0, 2]) / fps
    if duration <= 0:
        return 0.0
    speed = dist / duration  # px / s
    return float(np.clip(speed / speed_norm, 0.0, 1.0))


# ----------------------------------------------------------------------
# 综合健康评分
# ----------------------------------------------------------------------
POSTURE_WEIGHT = {
    "standing": 1.0,
    "lying_side": 0.75,
    "lying_belly": 0.60,
    "unknown": 0.50,
}


def compute_health_score(activity: float,
                         posture: str,
                         entropy: float = 0.0,
                         abnormal_ratio: float = 0.0) -> float:
    """综合健康评分 ∈ [0, 1]。

    H = 0.5·activity + 0.25·posture_w + 0.15·norm_entropy - 0.10·abnormal_ratio
    熵以 log2(3)≈1.585 归一化为 [0,1]。
    """
    p_w = POSTURE_WEIGHT.get(posture, 0.5)
    norm_entropy = entropy / 1.585
    score = (0.50 * activity
             + 0.25 * p_w
             + 0.15 * norm_entropy
             - 0.10 * abnormal_ratio)
    return float(np.clip(score, 0.0, 1.0))


# ----------------------------------------------------------------------
# 异常判别
# ----------------------------------------------------------------------
def flag_abnormal(weight_kg: float,
                  health_score: float,
                  herd_weight_mean: Optional[float] = None,
                  herd_weight_std: Optional[float] = None,
                  health_threshold: float = 0.40,
                  z_threshold: float = 2.0) -> List[str]:
    """返回异常标记列表。"""
    flags: List[str] = []
    if health_score < health_threshold:
        flags.append("LOW_HEALTH")
    if (herd_weight_mean is not None
            and herd_weight_std is not None
            and herd_weight_std > 0
            and weight_kg > 0):
        z = (weight_kg - herd_weight_mean) / herd_weight_std
        if abs(z) > z_threshold:
            flags.append("WEIGHT_OUTLIER")
    return flags


# ----------------------------------------------------------------------
# 单轨迹诊断（高级 API）
# ----------------------------------------------------------------------
def diagnose_trajectory(box_sizes: Sequence[Tuple[float, float, int]],
                        positions: Sequence[Tuple[float, float, int]],
                        fps: float,
                        frame_area: Optional[float] = None,
                        herd_stats: Optional[dict] = None) -> dict:
    """单条轨迹的完整诊断。"""
    weight = estimate_weight_kg(box_sizes, frame_area=frame_area)
    posture = compute_posture(box_sizes)
    entropy = posture_entropy(box_sizes)
    activity = compute_activity(positions, fps)
    health = compute_health_score(activity, posture, entropy=entropy)

    flags: List[str] = []
    if herd_stats:
        flags = flag_abnormal(weight, health,
                              herd_weight_mean=herd_stats.get("weight_mean"),
                              herd_weight_std=herd_stats.get("weight_std"))
    else:
        flags = flag_abnormal(weight, health)

    return {
        "weight_kg": round(weight, 2),
        "posture": posture,
        "posture_entropy": round(entropy, 3),
        "activity_score": round(activity, 3),
        "health_score": round(health, 3),
        "abnormal_flags": flags,
    }


# ----------------------------------------------------------------------
# 群体级别汇总
# ----------------------------------------------------------------------
def aggregate_herd(diagnoses: Iterable[dict]) -> dict:
    """对所有有效轨迹的诊断结果做群体汇总。"""
    diag_list = list(diagnoses)
    if not diag_list:
        return {
            "n": 0,
            "weight_mean": 0.0, "weight_std": 0.0,
            "weight_min": 0.0, "weight_max": 0.0,
            "health_mean": 0.0,
            "abnormal_count": 0,
            "low_health_count": 0,
            "weight_outlier_count": 0,
        }

    weights = np.array([d["weight_kg"] for d in diag_list], dtype=np.float32)
    healths = np.array([d["health_score"] for d in diag_list], dtype=np.float32)
    all_flags = [f for d in diag_list for f in d["abnormal_flags"]]

    return {
        "n": len(diag_list),
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "health_mean": float(healths.mean()),
        "abnormal_count": sum(1 for d in diag_list if d["abnormal_flags"]),
        "low_health_count": all_flags.count("LOW_HEALTH"),
        "weight_outlier_count": all_flags.count("WEIGHT_OUTLIER"),
    }


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # mock 一条轨迹：50 帧，框 100x80，每帧水平移动 3 px
    mock_box = [(100.0, 80.0, i) for i in range(50)]
    mock_pos = [(50 + 3 * i, 100, i) for i in range(50)]
    diag = diagnose_trajectory(mock_box, mock_pos, fps=25.0)
    print("[SELF-TEST] single trajectory diagnosis:")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    herd = aggregate_herd([diag, diag, diag])
    print("[SELF-TEST] herd aggregate (3 same pigs):")
    for k, v in herd.items():
        print(f"  {k}: {v}")
