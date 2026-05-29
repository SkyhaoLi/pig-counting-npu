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
# 井盖参照 & 透视校正配置
# ----------------------------------------------------------------------
_GRATE_CONFIG = {
    "enabled": False,
    "bbox": None,       # (x1, y1, x2, y2) 井盖像素坐标
    "size_m": 0.6,      # 井盖边长（米）
    "horizon_y": None,  # 消失线 Y（None → 自动取画面高度 30%）
}
_FRAME_HEIGHT = None


def set_grate_config(enabled, bbox=None, size_m=0.6, horizon_y=None, frame_height=None):
    """注入井盖参照配置。"""
    global _GRATE_CONFIG, _FRAME_HEIGHT
    _GRATE_CONFIG["enabled"] = enabled
    if bbox is not None:
        _GRATE_CONFIG["bbox"] = tuple(bbox)
    _GRATE_CONFIG["size_m"] = size_m
    if horizon_y is not None:
        _GRATE_CONFIG["horizon_y"] = horizon_y
    if frame_height is not None:
        _FRAME_HEIGHT = frame_height


# ----------------------------------------------------------------------
# 体重估计
# ----------------------------------------------------------------------
def estimate_weight_kg(box_sizes: Sequence[Tuple[float, float, int]],
                       weight_scale: float = 800.0,
                       frame_area: Optional[float] = None,
                       roi_image: "Optional[np.ndarray]" = None,
                       box_bottom_ys: Optional[Sequence[Tuple[float, int]]] = None,
                       frame_median_areas: Optional[Sequence[Tuple[float, int]]] = None) -> float:
    """估计体重（kg）。

    优先策略：注入了 `_WEIGHT_MODEL` 且提供了 `roi_image` 时走训练好的模型，
    否则回退到 box 面积启发式。

    启发式分支支持两种校正：
    1. 帧内归一化（frame_median_areas）：消除同帧猪只远近差异
    2. 井盖透视校正（box_bottom_ys + 井盖配置）：基于物理参照的距离补偿

    Args:
        box_sizes: [(w, h, frame_idx), ...] 序列（启发式用）
        weight_scale: 像素面积到 kg 的折算系数（启发式用，经验默认 800.0）
        frame_area: 如果给出，会做简单透视校正（启发式用，兜底）
        roi_image: 单张猪体 ROI 裁剪 (HxWx3 BGR uint8)；走模型分支时必须提供
        box_bottom_ys: [(y_bottom, frame_idx), ...] 每框底部 Y 坐标（透视校正用）
        frame_median_areas: [(median_area, frame_idx), ...] 每帧所有 bbox 面积中位数

    Returns:
        体重 kg；若两条路径都无数据则返回 0.0
    """
    # 模型分支
    if _WEIGHT_MODEL is not None and roi_image is not None and roi_image.size > 0:
        try:
            kg = float(_WEIGHT_MODEL.predict(roi_image))
            return float(np.clip(kg, 5.0, 350.0))
        except Exception as e:
            print(f"[health_module] weight model failed, falling back to heuristic: {e}")

    # 启发式分支
    if not box_sizes:
        return 0.0

    areas = np.array([w * h for w, h, _ in box_sizes], dtype=np.float32)
    median_area = float(np.median(areas))

    # 1. 帧内归一化：同一帧的猪距离摄像头差不多，用帧中位面积消除远近差异
    if frame_median_areas:
        fma_values = np.array([v for v, _ in frame_median_areas], dtype=np.float32)
        avg_frame_median = float(np.mean(fma_values))
        ref_area = 40000.0
        if avg_frame_median > 1:
            median_area = median_area * (ref_area / avg_frame_median)

    # 2. 井盖透视校正：利用井盖物理尺寸做距离补偿
    if _GRATE_CONFIG["enabled"] and _GRATE_CONFIG["bbox"] is not None and box_bottom_ys:
        bbox = _GRATE_CONFIG["bbox"]
        grate_cy = (bbox[1] + bbox[3]) / 2.0
        h_y = _GRATE_CONFIG.get("horizon_y")
        if h_y is None:
            h_y = _FRAME_HEIGHT * 0.3 if _FRAME_HEIGHT else 0

        bottom_values = np.array([y for y, _ in box_bottom_ys], dtype=np.float32)
        median_bottom_y = float(np.median(bottom_values))

        grate_dist = grate_cy - h_y
        pig_dist = median_bottom_y - h_y
        if grate_dist > 0 and pig_dist > 0:
            dist_ratio = pig_dist / grate_dist
            median_area = median_area * (dist_ratio ** 2)

    # 3. 兜底：旧式简单透视校正（无井盖时仍可用）
    elif frame_area and frame_area > 0 and not frame_median_areas:
        ref = frame_area * 0.05
        median_area = median_area * (ref / max(median_area, 1.0)) ** 0.2

    weight_kg = median_area / max(weight_scale, 1e-3)
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
                        herd_stats: Optional[dict] = None,
                        roi_image: "Optional[np.ndarray]" = None,
                        box_bottom_ys: Optional[Sequence[Tuple[float, int]]] = None,
                        frame_median_areas: Optional[Sequence[Tuple[float, int]]] = None,
                        class_id: int = 0) -> dict:
    """单条轨迹的完整诊断。

    Args:
        roi_image: 可选 — 该轨迹的代表性 ROI 裁剪 (BGR HxWx3 uint8)，
            注入了体重模型时用它做模型推理；为 None 时仅启发式。
        box_bottom_ys: [(y_bottom, frame_idx), ...] 每框底部 Y 坐标
        frame_median_areas: [(median_area, frame_idx), ...] 每帧 bbox 面积中位数
        class_id: 0=pig, 1=sheep（影响体重估算参数）
    """
    class_names = {0: 'pig', 1: 'sheep'}
    class_name = class_names.get(class_id, 'pig')

    # 羊不用体重模型（模型只训练了猪），走启发式
    use_roi = roi_image if class_id == 0 else None
    # 羊体型小，同样像素面积对应更轻体重 → weight_scale 更大
    weight_scale = 800.0 if class_id == 0 else 1200.0

    weight = estimate_weight_kg(box_sizes, weight_scale=weight_scale,
                                frame_area=frame_area,
                                roi_image=use_roi,
                                box_bottom_ys=box_bottom_ys,
                                frame_median_areas=frame_median_areas)
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
        "class_id": class_id,
        "class_name": class_name,
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
    print("[SELF-TEST] single trajectory diagnosis (no correction):")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # 测试帧内归一化
    mock_bottom_ys = [(300.0, i) for i in range(50)]
    mock_frame_med = [(8000.0, i) for i in range(50)]
    diag_norm = diagnose_trajectory(mock_box, mock_pos, fps=25.0,
                                    box_bottom_ys=mock_bottom_ys,
                                    frame_median_areas=mock_frame_med)
    print("[SELF-TEST] with frame normalization:")
    for k, v in diag_norm.items():
        print(f"  {k}: {v}")

    # 测试井盖透视校正
    set_grate_config(True, bbox=(100, 400, 220, 520), size_m=0.6, frame_height=720)
    diag_grate = diagnose_trajectory(mock_box, mock_pos, fps=25.0,
                                     box_bottom_ys=mock_bottom_ys,
                                     frame_median_areas=mock_frame_med)
    print("[SELF-TEST] with grate perspective correction:")
    for k, v in diag_grate.items():
        print(f"  {k}: {v}")

    # 关闭井盖，恢复默认
    set_grate_config(False)

    # 测试羊类别（不同的 weight_scale）
    diag_sheep = diagnose_trajectory(mock_box, mock_pos, fps=25.0, class_id=1)
    print("[SELF-TEST] sheep diagnosis (class_id=1):")
    for k, v in diag_sheep.items():
        print(f"  {k}: {v}")

    herd = aggregate_herd([diag, diag, diag])
    print("[SELF-TEST] herd aggregate (3 same pigs):")
    for k, v in herd.items():
        print(f"  {k}: {v}")

    # 混合群体
    herd_mixed = aggregate_herd([diag, diag_sheep])
    print("[SELF-TEST] herd aggregate (1 pig + 1 sheep):")
    for k, v in herd_mixed.items():
        print(f"  {k}: {v}")
