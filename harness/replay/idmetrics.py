"""Small identity metrics against ground-truth tracks. For publishable numbers use TrackEval
(HOTA, IDF1) via `replay.motformat export-tracks`; these exist so failure modes are unit-testable.

  id_switches    a ground-truth object changes predicted ID between consecutive matched frames
  id_transfers   one predicted ID is reused across different ground-truth objects
                 (the lead-to-follower handover that merges two visits into one)
"""
from __future__ import annotations

from collections import defaultdict

from .schema import TrackBox
from .trackers.greedy_iou import iou


def identity_report(gt: list[TrackBox], pred: list[TrackBox], iou_min: float = 0.5) -> dict:
    g, p = defaultdict(list), defaultdict(list)
    for b in gt:
        g[b.frame].append(b)
    for b in pred:
        p[b.frame].append(b)
    last: dict[int, int] = {}
    owners: dict[int, set] = defaultdict(set)
    switches = matched = 0
    for f in sorted(g):
        pairs = sorted(((iou(a.bbox, b.bbox), a.track_id, b.track_id) for a in g[f] for b in p.get(f, [])), reverse=True)
        ug, up = set(), set()
        for s, gid, pid in pairs:
            if s < iou_min:
                break
            if gid in ug or pid in up:
                continue
            ug.add(gid); up.add(pid)
            matched += 1
            owners[pid].add(gid)
            if gid in last and last[gid] != pid:
                switches += 1
            last[gid] = pid
    return {"gt_objects": len({b.track_id for b in gt}), "pred_ids": len({b.track_id for b in pred}),
            "id_switches": switches, "id_transfers": sum(len(v) - 1 for v in owners.values()),
            "gt_boxes_matched_pct": round(100 * matched / max(len(gt), 1), 1)}
