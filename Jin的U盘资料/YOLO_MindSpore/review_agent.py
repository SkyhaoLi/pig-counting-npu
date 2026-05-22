#!/usr/bin/env python3
"""Backward-compatible wrapper — delegates to PigCountingAgent."""

from __future__ import annotations

from pathlib import Path

from pig_counting_agent import PigCountingAgent


class HumanReviewAgent(PigCountingAgent):
    """Backward-compatible alias for the review subset of PigCountingAgent."""

    def __init__(self, registry_path):
        super().__init__(registry_path=registry_path)
