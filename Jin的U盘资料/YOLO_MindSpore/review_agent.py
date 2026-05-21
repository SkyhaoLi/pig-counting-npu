#!/usr/bin/env python3
"""Human-in-the-loop review agent for offline batch evaluation."""

from __future__ import annotations

from pathlib import Path

from human_review import apply_review, load_review_registry, summarize_results


class HumanReviewAgent:
    """Wrap review-registry logic behind an agent-style interface."""

    def __init__(self, registry_path):
        self.registry_path = Path(registry_path)
        self.registry = load_review_registry(self.registry_path)

    def assess(self, video_name, actual, detected_total):
        result = apply_review(self.registry, video_name, actual, detected_total)
        if actual is None and result["review_status"] == "accepted":
            result["review_status"] = "no_ground_truth"
            result["needs_review"] = 0
        return result

    def summarize(self, results):
        return summarize_results(results)
