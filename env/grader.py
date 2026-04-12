from __future__ import annotations

import math
import sqlite3
from typing import Any


# Keep scores visibly away from 0 and 1 so validator rounding never turns a
# valid score into 0.00 or 1.00.
MIN_SCORE = 0.01
MAX_SCORE = 0.99

CORRECTNESS_REWARD = 0.6
MAX_PERFORMANCE_REWARD = 0.2
VALID_SQL_BONUS = 0.1
STEP_PENALTY = 0.05
MAX_PENALIZED_STEPS = 16
MAX_INDEX_COUNT = 3


def _clamp_score(value: float) -> float:
    if not math.isfinite(value):
        return MIN_SCORE
    return max(MIN_SCORE, min(MAX_SCORE, round(float(value), 6)))


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (int, str)) or value is None:
        return value
    return str(value)


def normalize_rows(
    rows: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]],
) -> list[tuple[Any, ...]]:
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


def _plan_detail(row: Any) -> str:
    if isinstance(row, dict):
        value = row.get("detail", row)
    elif isinstance(row, (list, tuple)) and row:
        value = row[-1]
    else:
        value = row
    return str(value).upper()


def extract_plan_signals(
    plan_rows: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]] | None,
) -> dict[str, int]:
    if not plan_rows:
        return {"scan_count": 0, "index_count": 0}

    scan_count = 0
    index_count = 0

    for row in plan_rows:
        detail = _plan_detail(row)
        if "SCAN" in detail:
            scan_count += 1
        if "USING INDEX" in detail or "USING COVERING INDEX" in detail:
            index_count += 1

    return {"scan_count": scan_count, "index_count": index_count}


def compute_performance_reward(
    baseline_plan: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]] | None,
    candidate_plan: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]] | None,
) -> float:
    baseline = extract_plan_signals(baseline_plan)
    candidate = extract_plan_signals(candidate_plan)

    baseline_scan_count = max(baseline["scan_count"], 1)
    candidate_scan_count = min(candidate["scan_count"], baseline_scan_count)
    scan_improvement = (baseline_scan_count - candidate_scan_count) / baseline_scan_count

    index_bonus = min(candidate["index_count"], MAX_INDEX_COUNT) / MAX_INDEX_COUNT
    performance_score = MAX_PERFORMANCE_REWARD * ((0.7 * scan_improvement) + (0.3 * index_bonus))
    return round(max(0.0, min(MAX_PERFORMANCE_REWARD, performance_score)), 6)


def grade_submission(
    conn: sqlite3.Connection,
    sql: str,
    expected_rows: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]],
    baseline_plan: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]] | None,
    step_count: int,
) -> tuple[float, dict[str, Any]]:
    info: dict[str, Any] = {
        "correctness": 0.0,
        "performance": 0.0,
        "efficiency_penalty": 0.0,
        "validity_bonus": 0.0,
    }

    try:
        result_rows = conn.execute(sql).fetchall()
    except sqlite3.Error as exc:
        info["error"] = str(exc)
        final_score = _clamp_score(0.05)
        info["raw_score"] = 0.05
        info["final_reward"] = final_score
        return final_score, info

    try:
        result_norm = normalize_rows([tuple(row) for row in result_rows])
        expected_norm = normalize_rows(expected_rows)
        if result_norm == expected_norm:
            info["correctness"] = CORRECTNESS_REWARD

        candidate_plan = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
        info["performance"] = compute_performance_reward(baseline_plan, candidate_plan)
    except sqlite3.Error as exc:
        info["error"] = str(exc)

    info["validity_bonus"] = VALID_SQL_BONUS

    capped_steps = max(0, min(int(step_count), MAX_PENALIZED_STEPS))
    info["efficiency_penalty"] = round(STEP_PENALTY * capped_steps, 6)

    raw_score = (
        info["correctness"]
        + info["performance"]
        + info["validity_bonus"]
        - info["efficiency_penalty"]
    )
    final_score = _clamp_score(raw_score)

    info["raw_score"] = round(raw_score, 6)
    info["final_reward"] = final_score
    return final_score, info
