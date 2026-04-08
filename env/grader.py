from __future__ import annotations

import math
import sqlite3
from typing import Any


STRICT_SCORE_EPSILON = 1e-9  # Use tiny epsilon — just enough to exclude 0.0 and 1.0


def _strict_unit_interval(value: float) -> float:
    return max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, value))


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (int, str)) or value is None:
        return value
    return str(value)


def normalize_rows(rows: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    normalized: list[tuple[Any, ...]] = []

    for row in rows:
        if isinstance(row, dict):
            values = tuple(_normalize_scalar(row[key]) for key in sorted(row.keys()))
        elif isinstance(row, (list, tuple)):
            values = tuple(_normalize_scalar(item) for item in row)
        else:
            values = (_normalize_scalar(row),)
        normalized.append(values)

    normalized.sort()
    return normalized


def extract_plan_signals(plan_rows: list[tuple[Any, ...]] | list[list[Any]] | None) -> dict[str, int]:
    if not plan_rows:
        return {"scan_count": 0, "index_count": 0}

    scan_count = 0
    index_count = 0

    for row in plan_rows:
        detail = ""
        if isinstance(row, (tuple, list)) and row:
            detail = str(row[-1]).upper()
        else:
            detail = str(row).upper()

        if "SCAN" in detail:
            scan_count += 1
        if "USING INDEX" in detail or "USING COVERING INDEX" in detail:
            index_count += 1

    return {"scan_count": scan_count, "index_count": index_count}


def compute_performance_reward(
    baseline_plan: list[tuple[Any, ...]] | list[list[Any]] | None,
    candidate_plan: list[tuple[Any, ...]] | list[list[Any]] | None,
) -> float:
    baseline = extract_plan_signals(baseline_plan)
    candidate = extract_plan_signals(candidate_plan)

    baseline_scan = max(baseline["scan_count"], 1)
    candidate_scan = candidate["scan_count"]

    scan_reduction_ratio = (baseline_scan - min(candidate_scan, baseline_scan)) / baseline_scan
    index_bonus_ratio = min(candidate["index_count"], 3) / 3.0

    score_ratio = max(0.0, min(1.0, 0.7 * scan_reduction_ratio + 0.3 * index_bonus_ratio))
    return round(0.2 * score_ratio, 6)


def grade_submission(
    conn: sqlite3.Connection,
    sql: str,
    expected_rows: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]],
    baseline_plan: list[tuple[Any, ...]] | list[list[Any]] | None,
    step_count: int,
) -> tuple[float, dict[str, Any]]:
    info: dict[str, Any] = {
        "correctness": 0.0,
        "performance": 0.0,
        "efficiency_penalty": 0.0,
        "validity_bonus": 0.0,
    }

    try:
        cursor = conn.execute(sql)
        result_rows = cursor.fetchall()
    except sqlite3.Error as exc:
        info["error"] = str(exc)
        # On error, return a safe low score strictly inside (0, 1)
        safe_score = _strict_unit_interval(0.05)
        info["final_reward"] = safe_score
        return round(safe_score, 6), info

    result_norm = normalize_rows([tuple(row) for row in result_rows])
    expected_norm = normalize_rows(expected_rows)

    correctness = 0.6 if result_norm == expected_norm else 0.0
    info["correctness"] = correctness

    candidate_plan = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    performance = compute_performance_reward(baseline_plan, candidate_plan)
    info["performance"] = performance

    efficiency_penalty = 0.05 * step_count
    info["efficiency_penalty"] = efficiency_penalty

    validity_bonus = 0.1
    info["validity_bonus"] = validity_bonus

    raw_score = correctness + performance + validity_bonus - efficiency_penalty

    # Single, clean clamp strictly inside (0, 1) — no 0.0, no 1.0 allowed
    final_score = _strict_unit_interval(raw_score)

    info["final_reward"] = final_score
    return round(final_score, 6), info