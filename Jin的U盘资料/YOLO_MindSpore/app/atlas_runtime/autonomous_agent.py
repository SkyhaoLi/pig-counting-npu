#!/usr/bin/env python3
"""Backward-compatible wrapper — delegates to PigCountingAgent."""

from __future__ import annotations

import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.pig_counting_agent import PigCountingAgent


class AutonomousOpsAgent(PigCountingAgent):
    """Backward-compatible alias for the ops-monitoring subset of PigCountingAgent."""

    def __init__(self, log_dir=None, stale_frame_seconds=2.5,
                 reconnect_failure_threshold=20, low_fps_threshold=4.0,
                 drift_threshold=6, event_capacity=40, event_cooldown_seconds=5.0):
        super().__init__(
            registry_path=None,
            log_dir=log_dir,
            stale_frame_seconds=stale_frame_seconds,
            reconnect_failure_threshold=reconnect_failure_threshold,
            low_fps_threshold=low_fps_threshold,
            drift_threshold=drift_threshold,
            event_capacity=event_capacity,
            event_cooldown_seconds=event_cooldown_seconds,
        )
