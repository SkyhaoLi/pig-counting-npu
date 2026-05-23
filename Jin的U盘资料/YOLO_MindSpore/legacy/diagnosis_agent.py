#!/usr/bin/env python3
"""Backward-compatible wrapper — delegates to PigCountingAgent."""

from __future__ import annotations

from app.agents.pig_counting_agent import PigCountingAgent


class DiagnosisAgent(PigCountingAgent):
    """Backward-compatible alias for the diagnosis subset of PigCountingAgent."""

    def __init__(self, tracker_name="ByteTrack"):
        super().__init__(registry_path=None, tracker_name=tracker_name)
