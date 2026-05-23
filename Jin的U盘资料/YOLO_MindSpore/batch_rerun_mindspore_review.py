#!/usr/bin/env python3
"""Run offline batch processing with the local MindSpore fallback chain."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

from diagnosis_agent import DiagnosisAgent
from review_agent import HumanReviewAgent


SCRIPT_DIR = Path(__file__).parent
DEFAULT_MS_PROJECT = Path(r"C:\Users\Skyha\Documents\YOLO_MindSpore_restart")
DEFAULT_TRACK_SCRIPT = DEFAULT_MS_PROJECT / "track_and_count_mindspore.py"
DEFAULT_CONFIG = DEFAULT_MS_PROJECT / "mindyolo" / "configs" / "yolov8n_pig.yaml"
DEFAULT_WEIGHT = DEFAULT_MS_PROJECT / "tmp_deploy" / "weights" / "EMA_yolov8n_pig-10_1235.ckpt"
DEFAULT_PYTHON = Path(r"C:\Users\Skyha\.conda\envs\pig_count\python.exe")
REVIEW_REGISTRY = SCRIPT_DIR / "review_registry.json"

GROUP_TO_VIDEO_DIR = {
    "4": SCRIPT_DIR / "数据集" / "四",
    "5": SCRIPT_DIR / "数据集" / "五",
    "6": SCRIPT_DIR / "数据集" / "六",
}

GROUP_TO_OUTPUT_DIR = {
    "4": Path(r"C:\Users\Skyha\Desktop\pig_couter\output\batch_rerun_group4_mindspore_review"),
    "5": Path(r"C:\Users\Skyha\Desktop\pig_couter\output\batch_rerun_group5_mindspore_review"),
    "6": Path(r"C:\Users\Skyha\Desktop\pig_couter\output\batch_rerun_group6_mindspore_review"),
}


def detect_output_count(output_dir):
    summary = output_dir / "ByteTrack_summary.csv"
    if summary.exists():
        with open(summary, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            row = rows[0]
            return int(row["total_line"]), int(row["valid_traj"])

    report = output_dir / "ByteTrack_trajectory_report.csv"
    if report.exists():
        with open(report, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        for row in rows:
            if row and row[0] == "TOTAL COUNT":
                total_line = int(row[1])
            if row and row[0] == "TOTAL VALID (轨迹验证)":
                valid_traj = int(row[1])
                return total_line, valid_traj
    return None, None


def main():
    parser = argparse.ArgumentParser(description="MindSpore fallback batch rerun with review-aware outputs")
    parser.add_argument("--group", required=True, choices=sorted(GROUP_TO_VIDEO_DIR))
    parser.add_argument("--python", type=str, default=str(DEFAULT_PYTHON))
    parser.add_argument("--track_script", type=str, default=str(DEFAULT_TRACK_SCRIPT))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--weight", type=str, default=str(DEFAULT_WEIGHT))
    parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--output_base", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    python_exe = Path(args.python)
    track_script = Path(args.track_script)
    config_path = Path(args.config)
    weight_path = Path(args.weight)
    video_dir = GROUP_TO_VIDEO_DIR[args.group]
    output_base = Path(args.output_base) if args.output_base else GROUP_TO_OUTPUT_DIR[args.group]

    for path in [python_exe, track_script, config_path, weight_path, video_dir]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_base.mkdir(parents=True, exist_ok=True)
    review_agent = HumanReviewAgent(REVIEW_REGISTRY)
    diagnosis_agent = DiagnosisAgent()

    if args.group == "5":
        videos = sorted(video_dir.glob("*.mp4"), key=lambda p: p.stem)
    else:
        videos = sorted(video_dir.glob("*.mp4"), key=lambda p: int(re.match(r"(\d+)", p.stem).group(1)))

    results = []
    for index, video in enumerate(videos, 1):
        out_dir = output_base / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{index}/{len(videos)}] 处理: {video.name}")
        cmd = [
            str(python_exe), str(track_script),
            "--video_path", str(video),
            "--config", str(config_path),
            "--weight", str(weight_path),
            "--output_dir", str(out_dir),
            "--device", args.device,
            "--no_timestamp",
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])

        t0 = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = round(time.time() - t0, 1)
            total_line, valid_traj = detect_output_count(out_dir)
            m = re.search(r"-(\d+)头", video.stem)
            actual = int(m.group(1)) if m else None
            error = (total_line - actual) if (total_line is not None and actual is not None) else None
            review = review_agent.assess(video.name, actual, total_line)
            diagnosis = diagnosis_agent.analyze(
                out_dir,
                video.name,
                actual=actual,
                paper_actual=review["paper_actual"],
                total_line=total_line,
                valid_traj=valid_traj,
            )
            diagnosis_agent.write_reports(out_dir, diagnosis)
            results.append({
                "video": video.name,
                "actual": actual,
                "total_line": total_line,
                "valid_traj": valid_traj,
                "error_line": error,
                "paper_actual": review["paper_actual"],
                "paper_count": review["paper_count"],
                "paper_error": review["paper_error"],
                "paper_correct": review["paper_correct"],
                "review_status": review["review_status"],
                "reviewed": review["reviewed"],
                "needs_review": review["needs_review"],
                "review_note": review["review_note"],
                "review_tags": review["review_tags"],
                "diagnosis_primary_cause": diagnosis["primary_cause"],
                "diagnosis_secondary_causes": "|".join(diagnosis["secondary_causes"]),
                "diagnosis_confidence": diagnosis["diagnosis_confidence"],
                "diagnosis_window": "; ".join(
                    f"{w['label']}:{w['start_s']:.2f}-{w['end_s']:.2f}s" for w in diagnosis["suspect_windows"]
                ),
                "diagnosis_tracks": "|".join(str(t) for t in diagnosis["suspicious_track_ids"][:12]),
                "elapsed": elapsed,
            })
            print(f"  total_line={total_line} valid_traj={valid_traj} elapsed={elapsed}s")
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            print(f"  失败: {e}")
            results.append({
                "video": video.name,
                "actual": None,
                "total_line": None,
                "valid_traj": None,
                "error_line": None,
                "paper_actual": None,
                "paper_count": None,
                "paper_error": None,
                "paper_correct": None,
                "review_status": "run_failed",
                "reviewed": 0,
                "needs_review": 1,
                "review_note": str(e),
                "review_tags": "",
                "diagnosis_primary_cause": "run_failed",
                "diagnosis_secondary_causes": "",
                "diagnosis_confidence": "",
                "diagnosis_window": "",
                "diagnosis_tracks": "",
                "elapsed": elapsed,
            })

    out_csv = output_base / f"batch_rerun_group{args.group}_results.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video", "actual", "total_line", "valid_traj", "error_line",
                "paper_actual", "paper_count", "paper_error", "paper_correct",
                "review_status", "reviewed", "needs_review", "review_note",
                "review_tags", "diagnosis_primary_cause", "diagnosis_secondary_causes",
                "diagnosis_confidence", "diagnosis_window", "diagnosis_tracks", "elapsed",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    summary = review_agent.summarize(results)
    print(f"\n结果已保存: {out_csv}")
    print(
        f"人工复核: reviewed={summary['reviewed_count']} "
        f"pending={summary['pending_review_count']} "
        f"raw_errors={summary['raw_error_count']} "
        f"paper_errors={summary['paper_error_count']}"
    )


if __name__ == "__main__":
    main()
