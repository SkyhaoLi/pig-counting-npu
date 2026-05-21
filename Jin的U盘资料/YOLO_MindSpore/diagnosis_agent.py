#!/usr/bin/env python3
"""Diagnosis agent for analyzing counting outputs and inferring likely causes.

基于人工标注 error&reason 数据优化，覆盖以下 5 类真实错误模式：
1. ID丢失后重分配 —— 遮挡/拥挤导致目标消失，新ID过部分线多计 (G4-17)
2. 反向运动减计 —— 目标计数后反向跑回被线计数减去 (G2-38)
3. 遮挡致ID碎片化 —— 一个目标分配多个短命ID，全部无效致少计 (G4-15)
4. Ghost含完整子路径 —— 从OUT起始但内含完整 ENTRY->WAIT->OUT 穿越 (G2-36)
5. ENTRY区快速消失 —— 目标仅在ENTRY出现极短时间就消失/折返 (G3-11)
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


class DiagnosisAgent:
    """Read exported artifacts and infer likely issue positions and causes."""

    def __init__(self, tracker_name="ByteTrack"):
        self.tracker_name = tracker_name

    # ------------------------------------------------------------------
    # 核心分析
    # ------------------------------------------------------------------

    def analyze(self, output_dir, video_name, actual=None, paper_actual=None,
                total_line=None, valid_traj=None):
        output_dir = Path(output_dir)
        artifacts = self._collect_artifacts(output_dir)
        events = self._load_events(artifacts["events_csv"])
        trajectories, report_summary = self._load_trajectory_report(artifacts["trajectory_csv"])
        states = self._load_states(artifacts["state_txt"])
        summary = self._load_summary(artifacts["summary_csv"], report_summary)

        total_line = summary.get("total_line", total_line)
        valid_traj = summary.get("valid_traj", valid_traj)
        total_ids = summary.get("total_ids")

        raw_error = None if actual is None or total_line is None else total_line - actual
        paper_error = None if paper_actual is None or total_line is None else total_line - paper_actual

        # --- 分类轨迹 ---
        invalid_reasons = Counter()
        out_ghosts = []
        wait_ghosts = []
        retried_tracks = []
        stuck_wait_tracks = []
        high_recovery_tracks = []
        high_lost_tracks = []
        turned_back_tracks = []

        for row in trajectories:
            tid = int(row["TrackID"])
            reason = row["Reason"]
            invalid_reasons[reason] += 1
            state_meta = states.get(tid, {})

            if "Ghost (Started in OUT)" in reason:
                out_ghosts.append(tid)
            if "Ghost (Started in WAIT)" in reason:
                wait_ghosts.append(tid)
            if "Retried" in reason:
                retried_tracks.append(tid)
            if "Stuck in Wait" in reason:
                stuck_wait_tracks.append(tid)
            if "Turned Back" in reason:
                turned_back_tracks.append(tid)
            if state_meta.get("recovered_count", 0) >= 2:
                high_recovery_tracks.append(tid)
            if state_meta.get("lost_frames", 0) >= 5:
                high_lost_tracks.append(tid)

        traj_lookup = {int(r["TrackID"]): r for r in trajectories}
        event_by_tid = self._group_events_by_tid(events)

        # --- 新增：5 类深度检测 ---
        id_reassign = self._detect_id_reassignment(
            trajectories, states, events, event_by_tid, traj_lookup)
        reverse_deduct = self._detect_reverse_deduction(
            events, event_by_tid, traj_lookup, out_ghosts)
        fragmented = self._detect_id_fragmentation(
            trajectories, states, traj_lookup)
        ghost_valid = self._detect_ghost_with_valid_subpath(
            out_ghosts, traj_lookup)
        entry_vanish = self._detect_entry_vanish(
            turned_back_tracks, traj_lookup, states)

        early_out_events = [
            e for e in events
            if e["event"] == "NEW_ID" and e["zone"] == "OUT" and e["time_s"] <= 3.0
        ]

        # --- 组装诊断原因 ---
        causes = []
        evidence = []
        problem_tracks = []  # 汇总所有问题轨迹的详细信息

        if actual is not None and paper_actual is not None and actual != paper_actual:
            causes.append("标签定义与统计方向不一致")
            evidence.append(
                f"文件名真实值 {actual} 与人工复核真实值 {paper_actual} 不一致，"
                f"说明原始标签包含非统计方向目标。"
            )

        # [模式1] ID丢失后重分配导致多计
        if id_reassign["pairs"]:
            causes.append("ID丢失后重分配导致多计/少计")
            pairs_desc = "; ".join(
                f"ID{p['lost_id']}({p['lost_time']:.1f}s消失) -> "
                f"ID{p['new_id']}({p['new_time']:.1f}s出现, 过{p['new_lines_crossed']}条线)"
                for p in id_reassign["pairs"][:5]
            )
            evidence.append(
                f"检测到 {len(id_reassign['pairs'])} 对疑似ID重分配：{pairs_desc}。"
                f"部分新ID仅过部分线导致三线平均偏高。"
            )
            for p in id_reassign["pairs"]:
                problem_tracks.append({
                    "type": "ID重分配",
                    "track_ids": [p["lost_id"], p["new_id"]],
                    "time_range": f"{p['lost_time']:.1f}s - {p['new_time']:.1f}s",
                    "detail": f"ID{p['lost_id']}在{p['lost_time']:.1f}s消失，"
                              f"ID{p['new_id']}在{p['new_time']:.1f}s出现于{p['new_zone']}区，"
                              f"过了{p['new_lines_crossed']}条线",
                    "impact": "多计" if p["new_lines_crossed"] > 0 else "少计",
                })

        # [模式2] 反向运动导致线计数减少
        if reverse_deduct["tracks"]:
            causes.append("目标反向运动导致线计数被减")
            rev_desc = "; ".join(
                f"ID{t['tid']}在{t['reverse_time']:.1f}s反向"
                for t in reverse_deduct["tracks"][:5]
            )
            evidence.append(
                f"检测到 {len(reverse_deduct['tracks'])} 个目标计数后反向跑回：{rev_desc}。"
                f"线计数因反向穿越被减去，导致少计约 {reverse_deduct['estimated_loss']} 头。"
            )
            for t in reverse_deduct["tracks"]:
                problem_tracks.append({
                    "type": "反向运动减计",
                    "track_ids": [t["tid"]],
                    "time_range": f"{t['reverse_time']:.1f}s",
                    "detail": f"ID{t['tid']}完成正向穿越后在{t['reverse_time']:.1f}s反向运动，"
                              f"区域路径: {t['zone_history']}",
                    "impact": "少计",
                })

        # [模式3] 遮挡导致ID碎片化
        if fragmented["clusters"]:
            causes.append("遮挡导致ID碎片化，同一目标多个短命ID均无效")
            frag_desc = "; ".join(
                f"[{','.join(str(x) for x in c['ids'])}]在{c['time_range']}"
                for c in fragmented["clusters"][:3]
            )
            evidence.append(
                f"检测到 {len(fragmented['clusters'])} 组ID碎片簇：{frag_desc}。"
                f"同一时空区域出现多个极短轨迹(Ghost/TurnedBack)，"
                f"疑似同一目标被反复分配新ID但全部无效导致少计。"
            )
            for c in fragmented["clusters"]:
                problem_tracks.append({
                    "type": "ID碎片化",
                    "track_ids": c["ids"],
                    "time_range": c["time_range"],
                    "detail": f"ID组{c['ids']}在{c['time_range']}内出现于相近区域，"
                              f"单个持续<{c['max_duration']:.1f}s，均为无效轨迹",
                    "impact": "少计",
                })

        # [模式4] Ghost含完整子路径（从OUT开始但实际完成了完整穿越）
        if ghost_valid["tracks"]:
            causes.append("Ghost轨迹内含完整穿越子路径被误判为无效")
            gv_desc = "; ".join(
                f"ID{t['tid']}({t['subpath']})"
                for t in ghost_valid["tracks"][:5]
            )
            evidence.append(
                f"检测到 {len(ghost_valid['tracks'])} 个Ghost(OUT)轨迹内含完整"
                f"ENTRY->WAIT->OUT子路径：{gv_desc}。"
                f"这些轨迹的目标实际完成了有效穿越但因起始于OUT被判无效。"
            )
            for t in ghost_valid["tracks"]:
                problem_tracks.append({
                    "type": "Ghost含完整子路径",
                    "track_ids": [t["tid"]],
                    "time_range": f"{t['first_time']:.1f}s - {t['last_time']:.1f}s",
                    "detail": f"ID{t['tid']}从OUT起始，区域路径{t['zone_history']}，"
                              f"内含完整子路径{t['subpath']}",
                    "impact": "计数可能被低估(轨迹验证)或线计数已正确但轨迹验证偏低",
                })

        # [模式5] ENTRY区极短消失
        if entry_vanish["tracks"]:
            causes.append("目标在ENTRY区极短时间消失或折返")
            ev_desc = "; ".join(
                f"ID{t['tid']}(仅{t['duration']:.2f}s)"
                for t in entry_vanish["tracks"][:5]
            )
            evidence.append(
                f"检测到 {len(entry_vanish['tracks'])} 个目标仅在ENTRY区短暂出现后消失：{ev_desc}。"
                f"可能是速度过快/遮挡导致追踪丢失，或真实折返。"
            )
            for t in entry_vanish["tracks"]:
                problem_tracks.append({
                    "type": "ENTRY区快速消失",
                    "track_ids": [t["tid"]],
                    "time_range": f"{t['first_time']:.1f}s - {t['last_time']:.1f}s",
                    "detail": f"ID{t['tid']}仅在ENTRY存在{t['duration']:.2f}s，"
                              f"丢失帧{t['lost_frames']}，恢复{t['recovered']}次",
                    "impact": "少计(若为真实目标)" if t["duration"] > 0.5 else "正常折返(极短)",
                })

        # --- 保留原有检测（降低阈值使其与新检测互补）---
        if len(out_ghosts) >= 2 or len(early_out_events) >= 2:
            if "反向进入目标或非统计方向目标混入" not in [c for c in causes]:
                causes.append("反向进入目标或非统计方向目标混入")
                evidence.append(
                    f"Ghost(Started in OUT) 轨迹 {len(out_ghosts)} 个，"
                    f"前 3 秒在 OUT 区新生 ID {len(early_out_events)} 个。"
                )

        if high_recovery_tracks or high_lost_tracks:
            if "遮挡或重分配ID导致重复计数" not in [c for c in causes]:
                causes.append("遮挡或重分配ID导致重复计数")
                evidence.append(
                    f"高恢复轨迹(>=2次) {len(high_recovery_tracks)} 个，"
                    f"高丢失轨迹(>=5帧) {len(high_lost_tracks)} 个。"
                )

        if retried_tracks or stuck_wait_tracks:
            causes.append("入口区停留或折返导致计数波动")
            evidence.append(
                f"Retried 轨迹 {len(retried_tracks)} 个，"
                f"Stuck in Wait 轨迹 {len(stuck_wait_tracks)} 个。"
            )

        if total_line is not None and valid_traj is not None and abs(total_line - valid_traj) >= 2:
            causes.append("线计数与轨迹验证结果存在偏差")
            evidence.append(
                f"total_line={total_line}, valid_traj={valid_traj}, "
                f"gap={total_line - valid_traj}。"
            )

        # 三线不一致检测
        line_spread = self._detect_line_spread(summary)
        if line_spread:
            causes.append("三线计数不一致，部分目标仅过部分线")
            evidence.append(line_spread)

        if not causes:
            causes.append("未发现明显结构性问题")
            evidence.append("当前导出物中未出现足够强的异常模式。")

        # --- 时间窗口 ---
        windows = []
        windows.extend(self._build_windows("反向进入或方向冲突", out_ghosts, trajectories))
        windows.extend(self._build_windows(
            "遮挡与ID切换风险",
            high_recovery_tracks or high_lost_tracks, trajectories))
        windows.extend(self._build_windows(
            "停留/折返风险",
            retried_tracks or stuck_wait_tracks, trajectories))
        # 新增：为每类新检测的问题轨迹构建窗口
        if id_reassign["pairs"]:
            ids = []
            for p in id_reassign["pairs"]:
                ids.extend([p["lost_id"], p["new_id"]])
            windows.extend(self._build_windows("ID重分配风险", ids, trajectories))
        if fragmented["clusters"]:
            frag_ids = []
            for c in fragmented["clusters"]:
                frag_ids.extend(c["ids"])
            windows.extend(self._build_windows("ID碎片化", frag_ids, trajectories))
        if reverse_deduct["tracks"]:
            windows.extend(self._build_windows(
                "反向运动减计",
                [t["tid"] for t in reverse_deduct["tracks"]], trajectories))

        all_suspicious = sorted(set(
            out_ghosts + wait_ghosts + retried_tracks + stuck_wait_tracks
            + high_recovery_tracks + turned_back_tracks
            + [tid for p in id_reassign["pairs"] for tid in (p["lost_id"], p["new_id"])]
            + [tid for c in fragmented["clusters"] for tid in c["ids"]]
            + [t["tid"] for t in reverse_deduct["tracks"]]
            + [t["tid"] for t in ghost_valid["tracks"]]
            + [t["tid"] for t in entry_vanish["tracks"]]
        ))

        diagnosis = {
            "video": video_name,
            "artifacts": artifacts,
            "summary": summary,
            "actual": actual,
            "paper_actual": paper_actual,
            "raw_error": raw_error,
            "paper_error": paper_error,
            "primary_cause": causes[0],
            "secondary_causes": causes[1:],
            "evidence": evidence,
            "invalid_reason_counts": dict(invalid_reasons),
            "suspect_windows": windows[:8],
            "suspicious_track_ids": all_suspicious,
            "problem_tracks": problem_tracks,
            "diagnosis_confidence": self._confidence(causes, evidence, windows),
        }
        return diagnosis

    # ------------------------------------------------------------------
    # 报告输出
    # ------------------------------------------------------------------

    def write_reports(self, output_dir, diagnosis):
        output_dir = Path(output_dir)
        json_path = output_dir / f"{self.tracker_name}_diagnosis.json"
        md_path = output_dir / f"{self.tracker_name}_diagnosis.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(diagnosis, f, ensure_ascii=False, indent=2)

        lines = [
            f"# {diagnosis['video']} 诊断报告",
            "",
            f"- 主因：{diagnosis['primary_cause']}",
            f"- 次要原因：{', '.join(diagnosis['secondary_causes']) if diagnosis['secondary_causes'] else '无'}",
            f"- 诊断置信度：{diagnosis['diagnosis_confidence']}",
            "",
            "## 证据",
            "",
        ]
        for item in diagnosis["evidence"]:
            lines.append(f"- {item}")

        # 新增：问题轨迹详情表
        problem_tracks = diagnosis.get("problem_tracks", [])
        if problem_tracks:
            lines.extend(["", "## 问题轨迹详情", ""])
            lines.append("| 类型 | 轨迹ID | 时间段 | 影响 | 详情 |")
            lines.append("|------|--------|--------|------|------|")
            for pt in problem_tracks:
                ids_str = ",".join(str(x) for x in pt["track_ids"])
                lines.append(
                    f"| {pt['type']} | {ids_str} | {pt['time_range']} "
                    f"| {pt['impact']} | {pt['detail']} |"
                )

        lines.extend(["", "## 可疑时间窗口", ""])
        if diagnosis["suspect_windows"]:
            for window in diagnosis["suspect_windows"]:
                lines.append(
                    f"- {window['label']}：{window['start_s']:.2f}s - "
                    f"{window['end_s']:.2f}s，轨迹 {window['tracks']}"
                )
        else:
            lines.append("- 无明显可疑窗口")

        lines.extend(["", "## 汇总", ""])
        summary = diagnosis["summary"]
        lines.append(f"- total_line: {summary.get('total_line')}")
        lines.append(f"- valid_traj: {summary.get('valid_traj')}")
        lines.append(f"- total_ids: {summary.get('total_ids')}")
        lines.append(f"- raw_error: {diagnosis.get('raw_error')}")
        lines.append(f"- paper_error: {diagnosis.get('paper_error')}")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return json_path, md_path

    # ------------------------------------------------------------------
    # 5 类深度检测方法
    # ------------------------------------------------------------------

    @staticmethod
    def _group_events_by_tid(events):
        """按 track_id 分组事件"""
        by_tid = {}
        for e in events:
            by_tid.setdefault(e["track_id"], []).append(e)
        return by_tid

    def _detect_id_reassignment(self, trajectories, states, events,
                                event_by_tid, traj_lookup):
        """[模式1] 检测 ID 丢失后重分配。
        原理：一个 ID 在某时刻消失(last_frame)，很快一个新 ID 在 WAIT/OUT 区出现，
        且新 ID 是 Ghost(Started in WAIT/OUT)，说明可能是同一目标被重新分配了 ID。
        如 G4-17: ID25 在 6.8s 消失 -> ID42 在 9.13s 出现于 WAIT，过了 line 导致多计。
        """
        pairs = []
        # 找所有 Ghost(WAIT) 和短命 Ghost(OUT->WAIT->OUT) 轨迹作为候选 "新ID"
        candidate_new = []
        for row in trajectories:
            reason = row["Reason"]
            if "Ghost (Started in WAIT)" in reason or "Ghost (Started in OUT)" in reason:
                candidate_new.append(row)

        # 找最后消失的少量轨迹作为候选 "丢失ID"（只看有丢失帧或恢复记录的）
        # 同时限制：每个新Ghost只匹配时间最近的一个丢失ID
        claimed_new = set()
        for cand in sorted(candidate_new, key=lambda c: float(c["FirstTime(s)"])):
            new_tid = int(cand["TrackID"])
            if new_tid in claimed_new:
                continue
            new_first = float(cand["FirstTime(s)"])
            zone_hist = cand.get("ZoneHistory", "")
            zones = [z.strip() for z in zone_hist.split("->") if z.strip()]
            new_zone = zones[0] if zones else "UNKNOWN"
            lines_crossed = 0
            if "WAIT" in zones and "OUT" in zones:
                lines_crossed += 1
            if zones and zones[0] == "WAIT" and "OUT" in zones:
                lines_crossed += 1

            # 找消失时间最近的非Ghost轨迹
            best = None
            best_gap = 999
            for row in trajectories:
                tid = int(row["TrackID"])
                reason = row["Reason"]
                if "Ghost" in reason:
                    continue
                state_meta = states.get(tid, {})
                lost_time = float(row["LastTime(s)"])
                gap = new_first - lost_time
                # 限制: 0~3s 内消失，且丢失ID本身有丢失帧或高恢复
                if 0 < gap < 3.0 and gap < best_gap:
                    has_issue = (state_meta.get("lost_frames", 0) > 0
                                 or state_meta.get("recovered_count", 0) > 0)
                    if has_issue:
                        best = (tid, lost_time)
                        best_gap = gap
            if best:
                claimed_new.add(new_tid)
                pairs.append({
                    "lost_id": best[0],
                    "lost_time": best[1],
                    "new_id": new_tid,
                    "new_time": new_first,
                    "new_zone": new_zone,
                    "new_lines_crossed": lines_crossed,
                    "gap_s": round(best_gap, 2),
                })
        return {"pairs": pairs}

    def _detect_reverse_deduction(self, events, event_by_tid, traj_lookup,
                                  out_ghosts):
        """[模式2] 检测反向运动减计。
        原理：Ghost(OUT) 轨迹从 OUT->WAIT->ENTRY 表示目标反向跑回。
        如果该 ID 之前曾正向穿越过线(OUT方向)，线计数会因反向穿越被减去。
        如 G2-38: 8 个目标计数后反向跑回被减掉。
        """
        tracks = []
        for tid in out_ghosts:
            row = traj_lookup.get(tid)
            if not row:
                continue
            zone_hist = row.get("ZoneHistory", "")
            zones = [z.strip() for z in zone_hist.split("->") if z.strip()]
            # 检测 OUT->WAIT->ENTRY 模式（反向跑回）
            hist_str = "->".join(zones)
            if "OUT->WAIT->ENTRY" in hist_str:
                # 找到反向发生的时间
                state_changes = row.get("StateChanges", "")
                reverse_time = float(row["FirstTime(s)"])
                # 解析 state changes 找 WAIT->ENTRY 的时间
                for part in state_changes.split(";"):
                    part = part.strip()
                    if "WAIT->ENTRY" in part:
                        m = re.match(r"([\d.]+)s", part)
                        if m:
                            reverse_time = float(m.group(1))
                            break
                tracks.append({
                    "tid": tid,
                    "reverse_time": reverse_time,
                    "zone_history": hist_str,
                })
        return {
            "tracks": tracks,
            "estimated_loss": len(tracks),
        }

    def _detect_id_fragmentation(self, trajectories, states, traj_lookup):
        """[模式3] 检测 ID 碎片化。
        原理：同一时空窗口内出现多个极短(< 2s)的无效轨迹(Ghost/TurnedBack)，
        说明可能是同一目标因遮挡被反复分配新 ID。
        如 G4-15: ID54(0.7s), ID59(0s), ID60(1.1s) 都在 11.9s~14.8s 之间。
        """
        # 收集所有短命无效轨迹
        short_invalid = []
        for row in trajectories:
            is_valid = row.get("IsValid", "True")
            if str(is_valid).lower() == "true":
                continue
            duration = float(row.get("Duration(s)", 0))
            if duration <= 2.0:
                short_invalid.append({
                    "tid": int(row["TrackID"]),
                    "first": float(row["FirstTime(s)"]),
                    "last": float(row["LastTime(s)"]),
                    "duration": duration,
                    "reason": row.get("Reason", ""),
                })

        # 按时间窗口聚类（3s 内的归为一组）
        short_invalid.sort(key=lambda x: x["first"])
        clusters = []
        used = set()
        for i, a in enumerate(short_invalid):
            if a["tid"] in used:
                continue
            group = [a]
            used.add(a["tid"])
            for j in range(i + 1, len(short_invalid)):
                b = short_invalid[j]
                if b["tid"] in used:
                    continue
                if b["first"] - a["last"] < 3.0:
                    group.append(b)
                    used.add(b["tid"])
            if len(group) >= 2:
                clusters.append({
                    "ids": [g["tid"] for g in group],
                    "time_range": f"{group[0]['first']:.1f}s-{group[-1]['last']:.1f}s",
                    "max_duration": max(g["duration"] for g in group),
                })
        return {"clusters": clusters}

    def _detect_ghost_with_valid_subpath(self, out_ghosts, traj_lookup):
        """[模式4] 检测 Ghost(OUT) 内含完整 ENTRY->WAIT->OUT 子路径。
        如 G2-36: ID1 从 OUT 起始，完整路径含 ENTRY->WAIT->OUT 子序列。
        """
        tracks = []
        for tid in out_ghosts:
            row = traj_lookup.get(tid)
            if not row:
                continue
            zone_hist = row.get("ZoneHistory", "")
            zones = [z.strip() for z in zone_hist.split("->") if z.strip()]
            # 寻找 ENTRY->WAIT->OUT 子序列
            for i in range(len(zones) - 2):
                if zones[i] == "ENTRY" and zones[i + 1] == "WAIT" and zones[i + 2] == "OUT":
                    tracks.append({
                        "tid": tid,
                        "zone_history": zone_hist,
                        "subpath": "->".join(zones[i:i + 3]),
                        "first_time": float(row["FirstTime(s)"]),
                        "last_time": float(row["LastTime(s)"]),
                    })
                    break
        return {"tracks": tracks}

    def _detect_entry_vanish(self, turned_back_tracks, traj_lookup, states):
        """[模式5] 检测 ENTRY 区快速消失的轨迹。
        仅在 ENTRY 区出现极短时间后消失，可能是追踪丢失导致少计。
        如 G3-11: ID16 仅在 ENTRY 就反向跑走，少一头。
        """
        tracks = []
        for tid in turned_back_tracks:
            row = traj_lookup.get(tid)
            if not row:
                continue
            duration = float(row.get("Duration(s)", 0))
            state_meta = states.get(tid, {})
            zone_hist = row.get("ZoneHistory", "")
            # 仅在 ENTRY 区、且持续时间有一定长度（区分真实折返 vs 追踪丢失）
            if zone_hist.strip() == "ENTRY":
                tracks.append({
                    "tid": tid,
                    "duration": duration,
                    "first_time": float(row["FirstTime(s)"]),
                    "last_time": float(row["LastTime(s)"]),
                    "lost_frames": state_meta.get("lost_frames", 0),
                    "recovered": state_meta.get("recovered_count", 0),
                })
        return {"tracks": tracks}

    @staticmethod
    def _detect_line_spread(summary):
        """检测三线计数不一致。
        如 G4-17: line0=32, line1=32, line2=31，三线不同说明部分目标仅过部分线。
        """
        line0 = summary.get("line0")
        line1 = summary.get("line1")
        line2 = summary.get("line2")
        if line0 is None or line1 is None or line2 is None:
            return None
        spread = max(line0, line1, line2) - min(line0, line1, line2)
        if spread >= 2:
            return (
                f"三线计数 line0={line0}, line1={line1}, line2={line2}，"
                f"极差={spread}，说明有目标仅穿越了部分线。"
            )
        return None

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def _collect_artifacts(self, output_dir):
        tracker = self.tracker_name
        return {
            "result_video": str(output_dir / f"{tracker}_result.mp4"),
            "events_csv": str(output_dir / f"{tracker}_id_events.csv"),
            "state_txt": str(output_dir / f"{tracker}_state_changes.txt"),
            "summary_csv": str(output_dir / f"{tracker}_summary.csv"),
            "trajectory_csv": str(output_dir / f"{tracker}_trajectory_report.csv"),
        }

    def _load_events(self, path):
        path = Path(path)
        if not path.exists():
            return []
        rows = []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "frame": int(row["frame"]),
                    "timestamp": row["timestamp"],
                    "time_s": self._parse_seconds(row["timestamp"]),
                    "event": row["event"],
                    "track_id": int(row["track_id"]),
                    "zone": row["zone"],
                    "details": row["details"],
                })
        return rows

    def _load_trajectory_report(self, path):
        rows = []
        summary = {}
        path = Path(path)
        if not path.exists():
            return rows, summary
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                key = row[0]
                if key == "[SUMMARY]":
                    continue
                if key in {"Line0", "Line1", "Line2", "AvgRaw", "TOTAL COUNT", "TOTAL VALID (轨迹验证)", "TOTAL IDs"}:
                    summary[key] = row[1]
                    continue
                if header and len(row) >= len(header) and row[0].isdigit():
                    item = dict(zip(header, row[:len(header)]))
                    item["TrackID"] = int(item["TrackID"])
                    item["FirstTime(s)"] = float(item["FirstTime(s)"])
                    item["LastTime(s)"] = float(item["LastTime(s)"])
                    rows.append(item)
        parsed_summary = {
            "line0": self._maybe_int(summary.get("Line0")),
            "line1": self._maybe_int(summary.get("Line1")),
            "line2": self._maybe_int(summary.get("Line2")),
            "avg_raw": self._maybe_float(summary.get("AvgRaw")),
            "total_line": self._maybe_int(summary.get("TOTAL COUNT")),
            "valid_traj": self._maybe_int(summary.get("TOTAL VALID (轨迹验证)")),
            "total_ids": self._maybe_int(summary.get("TOTAL IDs")),
        }
        return rows, parsed_summary

    def _load_summary(self, path, fallback):
        path = Path(path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                row = rows[0]
                return {
                    "line0": self._maybe_int(row.get("line0")),
                    "line1": self._maybe_int(row.get("line1")),
                    "line2": self._maybe_int(row.get("line2")),
                    "total_line": self._maybe_int(row.get("total_line")),
                    "valid_traj": self._maybe_int(row.get("valid_traj")),
                    "total_ids": self._maybe_int(row.get("total_ids")),
                }
        return fallback

    def _load_states(self, path):
        states = {}
        path = Path(path)
        if not path.exists():
            return states

        current = None
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip()
                m = re.match(r"ID (\d+): .*?\((.+)\)", line)
                if m:
                    current = int(m.group(1))
                    states[current] = {"reason": m.group(2)}
                    continue
                if current is None:
                    continue
                if "首次出现:" in line:
                    states[current]["first_time"] = self._parse_parenthetical_seconds(line)
                elif "最后出现:" in line:
                    states[current]["last_time"] = self._parse_parenthetical_seconds(line)
                elif "丢失帧数:" in line:
                    states[current]["lost_frames"] = self._parse_trailing_int(line)
                elif "恢复次数:" in line:
                    states[current]["recovered_count"] = self._parse_trailing_int(line)
        return states

    def _build_windows(self, label, track_ids, trajectories):
        if not track_ids:
            return []
        lookup = {row["TrackID"]: row for row in trajectories}
        starts = []
        ends = []
        kept = []
        for tid in track_ids:
            row = lookup.get(tid)
            if not row:
                continue
            starts.append(float(row["FirstTime(s)"]))
            ends.append(float(row["LastTime(s)"]))
            kept.append(tid)
        if not starts:
            return []
        return [{
            "label": label,
            "start_s": max(0.0, min(starts) - 0.5),
            "end_s": max(ends) + 0.5,
            "tracks": kept[:10],
        }]

    def _confidence(self, causes, evidence, windows):
        score = 0.45 + min(0.2, len(causes) * 0.1) + min(0.2, len(windows) * 0.1) + min(0.15, len(evidence) * 0.03)
        return round(min(score, 0.95), 2)

    @staticmethod
    def _parse_seconds(text):
        text = (text or "").strip()
        if text.endswith("s"):
            text = text[:-1]
        try:
            return float(text)
        except ValueError:
            return 0.0

    @staticmethod
    def _parse_parenthetical_seconds(text):
        m = re.search(r"\(([\d.]+)s\)", text)
        return float(m.group(1)) if m else 0.0

    @staticmethod
    def _parse_trailing_int(text):
        m = re.search(r"(\d+)\s*$", text)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _maybe_int(value):
        if value in (None, "", "None"):
            return None
        return int(float(value))

    @staticmethod
    def _maybe_float(value):
        if value in (None, "", "None"):
            return None
        return float(value)
