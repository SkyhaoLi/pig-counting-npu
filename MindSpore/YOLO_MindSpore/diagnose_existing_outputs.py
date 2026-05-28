#!/usr/bin/env python3
"""Generate diagnosis reports for existing batch output directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from diagnosis_agent import DiagnosisAgent


def main():
    parser = argparse.ArgumentParser(description="Generate diagnosis reports for existing batch outputs")
    parser.add_argument("--results_csv", required=True, help="Existing results csv path")
    parser.add_argument("--output_base", required=True, help="Directory containing per-video output folders")
    parser.add_argument("--tracker_name", default="ByteTrack")
    parser.add_argument("--summary_name", default="diagnosis_summary.csv")
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    output_base = Path(args.output_base)
    agent = DiagnosisAgent(tracker_name=args.tracker_name)

    rows = list(csv.DictReader(open(results_csv, encoding="utf-8")))
    summary_rows = []

    for row in rows:
        video = row["video"]
        out_dir = output_base / Path(video).stem
        if not out_dir.exists():
            continue

        actual = int(row["actual"]) if row.get("actual") not in ("", None) else None
        paper_actual = int(row["paper_actual"]) if row.get("paper_actual") not in ("", None) else None
        total_line = int(row["total_line"]) if row.get("total_line") not in ("", None) else None
        valid_traj = int(row["valid_traj"]) if row.get("valid_traj") not in ("", None) else None

        diagnosis = agent.analyze(
            out_dir,
            video,
            actual=actual,
            paper_actual=paper_actual,
            total_line=total_line,
            valid_traj=valid_traj,
        )
        json_path, md_path = agent.write_reports(out_dir, diagnosis)

        summary_rows.append({
            "video": video,
            "primary_cause": diagnosis["primary_cause"],
            "secondary_causes": "|".join(diagnosis["secondary_causes"]),
            "diagnosis_confidence": diagnosis["diagnosis_confidence"],
            "diagnosis_window": "; ".join(
                f"{w['label']}:{w['start_s']:.2f}-{w['end_s']:.2f}s" for w in diagnosis["suspect_windows"]
            ),
            "diagnosis_tracks": "|".join(str(t) for t in diagnosis["suspicious_track_ids"][:15]),
            "diagnosis_json": str(json_path),
            "diagnosis_md": str(md_path),
        })

    summary_path = output_base / args.summary_name
    if summary_rows:
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(summary_path)


if __name__ == "__main__":
    main()
