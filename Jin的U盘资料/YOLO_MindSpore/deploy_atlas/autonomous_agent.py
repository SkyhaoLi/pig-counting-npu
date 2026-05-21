#!/usr/bin/env python3
"""Autonomous operations agent for the realtime pig counting service."""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path


class AutonomousOpsAgent:
    """Monitor stream health and request bounded recovery actions."""

    def __init__(
        self,
        log_dir=None,
        stale_frame_seconds=2.5,
        reconnect_failure_threshold=20,
        low_fps_threshold=4.0,
        drift_threshold=6,
        event_capacity=40,
        event_cooldown_seconds=5.0,
    ):
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.log_dir / "ops_events.jsonl"
        else:
            self.log_path = None

        self.stale_frame_seconds = stale_frame_seconds
        self.reconnect_failure_threshold = reconnect_failure_threshold
        self.low_fps_threshold = low_fps_threshold
        self.drift_threshold = drift_threshold
        self.event_capacity = event_capacity
        self.event_cooldown_seconds = event_cooldown_seconds

        self.status = "BOOT"
        self.health_score = 100.0
        self.anomaly_count = 0
        self.recovery_count = 0
        self.reconnect_requests = 0
        self.pending_reconnect = False
        self.last_frame_wall_time = None
        self.last_total_count = 0
        self.events = deque(maxlen=event_capacity)
        self._last_emit_times = {}

    def _emit(self, kind, severity, message, **fields):
        now = time.time()
        cooldown_key = f"{kind}:{severity}"
        last_emit = self._last_emit_times.get(cooldown_key, 0.0)
        if severity != "info" and now - last_emit < self.event_cooldown_seconds:
            return
        self._last_emit_times[cooldown_key] = now

        event = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "severity": severity,
            "message": message,
            "fields": fields,
        }
        self.events.appendleft(event)
        if severity in ("warn", "error"):
            self.anomaly_count += 1
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def note_stream_started(self, source):
        self.status = "HEALTHY"
        self.health_score = 100.0
        self.pending_reconnect = False
        self.last_frame_wall_time = time.time()
        self._emit("stream_started", "info", "Stream opened", source=source)

    def note_frame(self, frame_idx, infer_fps, total_count, valid_traj, total_ids):
        now = time.time()
        self.last_frame_wall_time = now
        self.pending_reconnect = False

        drift = abs(total_count - valid_traj)
        score = 100.0
        status = "HEALTHY"

        if infer_fps < self.low_fps_threshold:
            score -= min(35.0, (self.low_fps_threshold - infer_fps) * 8.0)
            status = "WARN"
            self._emit(
                "low_fps",
                "warn",
                "Inference FPS dropped below threshold",
                frame=frame_idx,
                infer_fps=round(infer_fps, 2),
                threshold=self.low_fps_threshold,
            )

        if drift > self.drift_threshold:
            score -= min(30.0, (drift - self.drift_threshold) * 2.0)
            status = "WARN"
            self._emit(
                "count_drift",
                "warn",
                "Line-count and valid-trajectory counts diverged",
                frame=frame_idx,
                total_count=total_count,
                valid_traj=valid_traj,
                total_ids=total_ids,
            )

        if total_count < self.last_total_count:
            score -= 10.0
            status = "WARN"
            self._emit(
                "count_drop",
                "warn",
                "Total count decreased unexpectedly",
                frame=frame_idx,
                previous_total=self.last_total_count,
                current_total=total_count,
            )

        self.last_total_count = total_count
        self.status = status
        self.health_score = max(0.0, round(score, 1))

    def note_waiting_for_frame(self, wait_seconds, failure_streak):
        if wait_seconds < self.stale_frame_seconds:
            return

        self.status = "WARN"
        self.health_score = max(0.0, round(100.0 - min(60.0, wait_seconds * 10.0), 1))
        self._emit(
            "stale_frame",
            "warn",
            "No fresh frame available",
            wait_seconds=round(wait_seconds, 2),
            failure_streak=failure_streak,
        )

        if failure_streak >= self.reconnect_failure_threshold and not self.pending_reconnect:
            self.pending_reconnect = True
            self.reconnect_requests += 1
            self.status = "RECOVERING"
            self._emit(
                "reconnect_requested",
                "error",
                "Agent requested stream reconnect",
                failure_streak=failure_streak,
                wait_seconds=round(wait_seconds, 2),
            )

    def consume_actions(self):
        actions = {"reconnect": self.pending_reconnect}
        self.pending_reconnect = False
        return actions

    def note_reconnect_result(self, success, detail=""):
        if success:
            self.recovery_count += 1
            self.status = "HEALTHY"
            self.health_score = 100.0
            self._emit("reconnect_success", "info", "Stream reconnect succeeded", detail=detail)
        else:
            self.status = "ERROR"
            self.health_score = 0.0
            self._emit("reconnect_failed", "error", "Stream reconnect failed", detail=detail)

    def snapshot(self):
        return {
            "status": self.status,
            "health_score": self.health_score,
            "anomaly_count": self.anomaly_count,
            "recovery_count": self.recovery_count,
            "reconnect_requests": self.reconnect_requests,
            "latest_event": self.events[0]["message"] if self.events else "",
            "events": list(self.events)[:8],
        }
