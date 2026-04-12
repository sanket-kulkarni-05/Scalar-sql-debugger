from __future__ import annotations

import sqlite3
from typing import Any


MIN_SCORE = 0.01
MAX_SCORE = 0.99

EASY_CORRECTNESS = 0.8
EASY_VALIDITY = 0.1
EASY_EFFICIENCY = 0.05

MEDIUM_CORRECTNESS = 0.65
MEDIUM_PERFORMANCE = 0.15
MEDIUM_VALIDITY = 0.1
MEDIUM_EFFICIENCY = 0.05

HARD_CORRECTNESS = 0.6
HARD_PERFORMANCE = 0.2
HARD_VALIDITY = 0.1
HARD_EFFICIENCY = 0.05


def _clamp_score(value: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, round(float(value), 6)))


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (int, str)) or value is None:
        return value
    return str(value)


def _normalize_rows(
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


def _execute_query(
    conn: sqlite3.Connection,
    sql: str,
) -> tuple[list[tuple[Any, ...]] | None, str | None]:
    try:
        rows = conn.execute(sql).fetchall()
        return [tuple(row) for row in rows], None
    except sqlite3.Error as exc:
        return None, str(exc)


def _matches_expected(
    result_rows: list[tuple[Any, ...]],
    expected_rows: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]],
) -> bool:
    return _normalize_rows(result_rows) == _normalize_rows(expected_rows)


def _extract_plan_signals(
    plan_rows: list[tuple[Any, ...]] | list[list[Any]] | None,
) -> dict[str, int]:
    if not plan_rows:
        return {"scan_count": 0, "index_count": 0}

    scan_count = 0
    index_count = 0

    for row in plan_rows:
        detail = str(row[-1] if isinstance(row, (tuple, list)) and row else row).upper()
        if "SCAN" in detail:
            scan_count += 1
        if "USING INDEX" in detail or "USING COVERING INDEX" in detail:
            index_count += 1

    return {"scan_count": scan_count, "index_count": index_count}


def _performance_component(
    baseline_plan: list[tuple[Any, ...]] | list[list[Any]] | None,
    sql: str,
    conn: sqlite3.Connection,
    max_reward: float,
) -> float:
    try:
        candidate_plan = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    except sqlite3.Error:
        return 0.0

    baseline = _extract_plan_signals(baseline_plan)
    candidate = _extract_plan_signals(candidate_plan)

    baseline_scan_count = max(baseline["scan_count"], 1)
    candidate_scan_count = min(candidate["scan_count"], baseline_scan_count)
    scan_improvement = (baseline_scan_count - candidate_scan_count) / baseline_scan_count

    index_bonus = min(candidate["index_count"], 3) / 3.0
    score = max_reward * ((0.7 * scan_improvement) + (0.3 * index_bonus))
    return round(max(0.0, min(max_reward, score)), 6)


def _efficiency_component(step_count: int, max_reward: float, cutoff: int) -> float:
    return max_reward if int(step_count) <= cutoff else 0.0


def grade_easy(
    conn: sqlite3.Connection,
    sql: str,
    expected_rows: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]],
    step_count: int,
) -> tuple[float, dict[str, Any]]:
    result_rows, error = _execute_query(conn, sql)
    breakdown: dict[str, Any] = {
        "task_tier": "easy",
        "correctness": 0.0,
        "validity_bonus": 0.0,
        "efficiency_bonus": 0.0,
    }

    if error:
        breakdown["error"] = error
        score = _clamp_score(0.05)
        breakdown["final_reward"] = score
        return score, breakdown

    breakdown["validity_bonus"] = EASY_VALIDITY
    if result_rows is not None and _matches_expected(result_rows, expected_rows):
        breakdown["correctness"] = EASY_CORRECTNESS
    breakdown["efficiency_bonus"] = _efficiency_component(step_count, EASY_EFFICIENCY, cutoff=3)

    raw_score = sum(
        [
            breakdown["correctness"],
            breakdown["validity_bonus"],
            breakdown["efficiency_bonus"],
        ]
    )
    score = _clamp_score(raw_score)
    breakdown["raw_score"] = round(raw_score, 6)
    breakdown["final_reward"] = score
    return score, breakdown


def grade_medium(
    conn: sqlite3.Connection,
    sql: str,
    expected_rows: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]],
    baseline_plan: list[tuple[Any, ...]] | list[list[Any]] | None,
    step_count: int,
) -> tuple[float, dict[str, Any]]:
    result_rows, error = _execute_query(conn, sql)
    breakdown: dict[str, Any] = {
        "task_tier": "medium",
        "correctness": 0.0,
        "performance": 0.0,
        "validity_bonus": 0.0,
        "efficiency_bonus": 0.0,
    }

    if error:
        breakdown["error"] = error
        score = _clamp_score(0.05)
        breakdown["final_reward"] = score
        return score, breakdown

    breakdown["validity_bonus"] = MEDIUM_VALIDITY
    if result_rows is not None and _matches_expected(result_rows, expected_rows):
        breakdown["correctness"] = MEDIUM_CORRECTNESS
    breakdown["performance"] = _performance_component(baseline_plan, sql, conn, MEDIUM_PERFORMANCE)
    breakdown["efficiency_bonus"] = _efficiency_component(step_count, MEDIUM_EFFICIENCY, cutoff=5)

    raw_score = sum(
        [
            breakdown["correctness"],
            breakdown["performance"],
            breakdown["validity_bonus"],
            breakdown["efficiency_bonus"],
        ]
    )
    score = _clamp_score(raw_score)
    breakdown["raw_score"] = round(raw_score, 6)
    breakdown["final_reward"] = score
    return score, breakdown


def grade_hard(
    conn: sqlite3.Connection,
    sql: str,
    expected_rows: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]],
    baseline_plan: list[tuple[Any, ...]] | list[list[Any]] | None,
    step_count: int,
) -> tuple[float, dict[str, Any]]:
    result_rows, error = _execute_query(conn, sql)
    breakdown: dict[str, Any] = {
        "task_tier": "hard",
        "correctness": 0.0,
        "performance": 0.0,
        "validity_bonus": 0.0,
        "efficiency_bonus": 0.0,
    }

    if error:
        breakdown["error"] = error
        score = _clamp_score(0.05)
        breakdown["final_reward"] = score
        return score, breakdown

    breakdown["validity_bonus"] = HARD_VALIDITY
    if result_rows is not None and _matches_expected(result_rows, expected_rows):
        breakdown["correctness"] = HARD_CORRECTNESS
    breakdown["performance"] = _performance_component(baseline_plan, sql, conn, HARD_PERFORMANCE)
    breakdown["efficiency_bonus"] = _efficiency_component(step_count, HARD_EFFICIENCY, cutoff=6)

    raw_score = sum(
        [
            breakdown["correctness"],
            breakdown["performance"],
            breakdown["validity_bonus"],
            breakdown["efficiency_bonus"],
        ]
    )
    score = _clamp_score(raw_score)
    breakdown["raw_score"] = round(raw_score, 6)
    breakdown["final_reward"] = score
    return score, breakdown


def grade_submission(
    conn: sqlite3.Connection,
    sql: str,
    expected_rows: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]],
    baseline_plan: list[tuple[Any, ...]] | list[list[Any]] | list[dict[str, Any]] | None,
    step_count: int,
    difficulty: str = "hard",
) -> tuple[float, dict[str, Any]]:
    grader = GRADERS.get(difficulty, grade_hard)
    if grader is grade_easy:
        return grader(
            conn=conn,
            sql=sql,
            expected_rows=expected_rows,
            step_count=step_count,
        )
    return grader(
        conn=conn,
        sql=sql,
        expected_rows=expected_rows,
        baseline_plan=baseline_plan,
        step_count=step_count,
    )


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
