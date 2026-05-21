#!/usr/bin/env python3
"""Helpers for paper-oriented human review and special-case handling."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _video_match_key(name):
    stem = Path(name).stem
    match = re.match(r"^(\d+-\d+)", stem)
    if match:
        return match.group(1)
    return stem


def _find_review_entry(registry, video_name):
    videos = registry.get("videos", {})
    if video_name in videos:
        return videos[video_name]

    target_key = _video_match_key(video_name)
    for key, value in videos.items():
        if _video_match_key(key) == target_key:
            return value
    return {}


def load_review_registry(path):
    """Load review metadata from JSON, returning an empty registry if missing."""
    path = Path(path)
    if not path.exists():
        return {"schema_version": 1, "videos": {}}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("videos", {})
    if not isinstance(videos, dict):
        raise ValueError(f"Invalid review registry: {path}")

    return {
        "schema_version": data.get("schema_version", 1),
        "videos": videos,
    }


def apply_review(registry, video_name, actual, detected_total):
    """Return raw and paper-oriented metrics for a processed video."""
    review = _find_review_entry(registry, video_name)
    raw_error = None
    if actual is not None and detected_total is not None:
        raw_error = detected_total - actual

    result = {
        "raw_error": raw_error,
        "review_status": "accepted" if raw_error in (None, 0) else "needs_review",
        "paper_error": raw_error,
        "paper_actual": actual,
        "paper_count": detected_total,
        "review_note": "",
        "review_tags": "",
        "reviewed": 0,
        "needs_review": 0 if raw_error in (None, 0) else 1,
    }

    if review:
        result["reviewed"] = 1
        result["review_status"] = review.get("review_status", "human_reviewed")
        result["review_note"] = review.get("note", "")
        result["review_tags"] = "|".join(review.get("tags", []))

        if "paper_error" in review:
            result["paper_error"] = review["paper_error"]
        elif review.get("accept_as_correct"):
            result["paper_error"] = 0

        if "paper_actual" in review:
            result["paper_actual"] = review["paper_actual"]
        if "paper_count" in review:
            result["paper_count"] = review["paper_count"]
        if result["paper_actual"] is not None and result["paper_count"] is not None:
            result["paper_error"] = result["paper_count"] - result["paper_actual"]

        result["needs_review"] = 1 if result["review_status"] == "needs_review" else 0
    elif actual is None:
        result["review_status"] = "no_ground_truth"
        result["needs_review"] = 0

    paper_error = result["paper_error"]
    result["paper_correct"] = "" if paper_error is None else int(paper_error == 0)
    return result


def summarize_results(results):
    """Summaries used by batch scripts and paper tables."""
    raw_errors = [r for r in results if r.get("raw_error") not in (None, 0)]
    paper_errors = [r for r in results if r.get("paper_error") not in (None, 0)]
    pending = [r for r in results if r.get("needs_review")]
    reviewed = [r for r in results if r.get("reviewed")]
    return {
        "raw_error_count": len(raw_errors),
        "paper_error_count": len(paper_errors),
        "pending_review_count": len(pending),
        "reviewed_count": len(reviewed),
    }
