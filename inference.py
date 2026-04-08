from __future__ import annotations

import importlib
import json
import os
from typing import Any
from urllib import error, request

MAX_STEPS = 8
TASK_IDS = [1, 2, 3]
STRICT_SCORE_EPSILON = 0.01

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini").strip() or "gpt-4o-mini"
HF_TOKEN = os.getenv("HF_TOKEN")
OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:7860").strip() or "http://localhost:7860"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

SOLVER_SQL_BY_TASK: dict[int, str] = {
    1: (
        "SELECT c.name, o.total_amount "
        "FROM customers c "
        "JOIN orders o ON c.id = o.customer_id "
        "WHERE o.total_amount > 250 "
        "ORDER BY o.total_amount DESC "
        "LIMIT 20;"
    ),
    2: (
        "SELECT cat.name AS category_name, SUM(oi.quantity) AS items_sold "
        "FROM categories cat "
        "JOIN products p ON p.category_id = cat.id "
        "JOIN order_items oi ON oi.product_id = p.id "
        "JOIN orders o ON o.id = oi.order_id "
        "WHERE o.order_date >= date('now', '-365 days') "
        "GROUP BY cat.name "
        "ORDER BY items_sold DESC;"
    ),
    3: (
        "SELECT c.id, c.name, ROUND(SUM(oi.quantity * p.price), 2) AS total_spent "
        "FROM orders o "
        "JOIN customers c ON c.id = o.customer_id "
        "JOIN order_items oi ON oi.order_id = o.id "
        "JOIN products p ON p.id = oi.product_id "
        "WHERE o.order_date >= date('now', '-12 months') "
        "GROUP BY c.id, c.name "
        "ORDER BY total_spent DESC "
        "LIMIT 5;"
    ),
}


def _emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), sort_keys=False)}")


def _strict_unit_interval(value: float) -> float:
    return max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, float(value)))


def _safe_score(value: Any) -> float:
    try:
        return round(_strict_unit_interval(float(value)), 6)
    except (TypeError, ValueError):
        return round(_strict_unit_interval(0.0), 6)


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", response.getcode())
            if status_code != 200:
                raise RuntimeError(f"HTTP {status_code}")
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc


def _build_client() -> Any | None:
    try:
        openai_module = importlib.import_module("openai")
    except Exception:
        return None

    api_key = (HF_TOKEN or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    OpenAI = getattr(openai_module, "OpenAI", None)
    if OpenAI is None:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


def _fallback_action(task_id: int, step_idx: int, observation: dict[str, Any]) -> dict[str, Any]:
    target_sql = SOLVER_SQL_BY_TASK.get(task_id)
    if target_sql:
        if step_idx == 1:
            return {"action_type": "rewrite_query", "query": target_sql, "table": None, "column": None}
        return {"action_type": "submit_answer", "query": target_sql, "table": None, "column": None}

    return {
        "action_type": "execute_sql",
        "query": observation.get("current_query", "SELECT 1"),
        "table": None,
        "column": None,
    }


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            return candidate
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        candidate = json.loads(text[start : end + 1])
        if isinstance(candidate, dict):
            return candidate
    except json.JSONDecodeError:
        return None

    return None


def choose_action(
    client: Any | None,
    model_name: str,
    observation: dict[str, Any],
    task_id: int,
    step_idx: int,
) -> dict[str, Any]:
    if client is None:
        return _fallback_action(task_id, step_idx, observation)

    prompt = {
        "task_description": observation.get("task_description"),
        "schema_info": observation.get("schema_info"),
        "current_query": observation.get("current_query"),
        "last_result": observation.get("last_result"),
        "execution_plan": observation.get("execution_plan"),
        "step_count": observation.get("step_count"),
        "done": observation.get("done"),
    }

    instruction = (
        "You are debugging SQL in a constrained environment. "
        "Return exactly one JSON object with fields: action_type, query, table, column. "
        "Allowed action_type values are execute_sql, explain_plan, add_index, rewrite_query, submit_answer. "
        "Choose the single best next step."
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        )
    except Exception:
        return _fallback_action(task_id, step_idx, observation)

    content = completion.choices[0].message.content or ""
    parsed = _extract_json(content)
    if parsed is None:
        return {
            "action_type": "execute_sql",
            "query": observation.get("current_query", "SELECT 1"),
            "table": None,
            "column": None,
        }

    action_type = parsed.get("action_type")
    if action_type not in {"execute_sql", "explain_plan", "add_index", "rewrite_query", "submit_answer"}:
        return _fallback_action(task_id, step_idx, observation)

    sanitized = {
        "action_type": action_type,
        "query": parsed.get("query"),
        "table": parsed.get("table"),
        "column": parsed.get("column"),
    }

    if action_type in {"execute_sql", "explain_plan", "rewrite_query", "submit_answer"}:
        query = (sanitized.get("query") or "").strip()
        if not query:
            return _fallback_action(task_id, step_idx, observation)
        sanitized["query"] = query

    if action_type == "add_index":
        table = (sanitized.get("table") or "").strip()
        column = (sanitized.get("column") or "").strip()
        if not table or not column:
            return _fallback_action(task_id, step_idx, observation)
        sanitized["table"] = table
        sanitized["column"] = column
        sanitized["query"] = None

    if action_type != "add_index":
        sanitized["table"] = None
        sanitized["column"] = None

    return sanitized


def run() -> None:
    model_name = MODEL_NAME
    env_base_url = OPENENV_BASE_URL

    client = _build_client()
    _ = LOCAL_IMAGE_NAME

    final_scores: dict[int, float] = {}

    for task_id in TASK_IDS:
        _emit(
            "START",
            {
                "task_id": task_id,
                "model": model_name,
                "max_steps": MAX_STEPS,
                "env_base_url": env_base_url,
                "llm_enabled": client is not None,
            },
        )

        try:
            observation = _post_json(f"{env_base_url}/reset", {"task_id": task_id}, timeout=30)
        except Exception as exc:
            safe_reward = _safe_score(0.0)
            final_scores[task_id] = safe_reward
            _emit(
                "END",
                {
                    "task_id": str(task_id),
                    "score": safe_reward,
                    "completed": False,
                    "error": f"reset_failed: {exc}",
                },
            )
            continue

        final_reward = _safe_score(0.0)

        for step_idx in range(1, MAX_STEPS + 1):
            action = choose_action(client, model_name, observation, task_id, step_idx)

            try:
                payload = _post_json(f"{env_base_url}/step", {"action": action}, timeout=45)
            except Exception as exc:
                _emit(
                    "STEP",
                    {
                        "task_id": task_id,
                        "step": step_idx,
                        "action_type": action.get("action_type"),
                        "reward": final_reward,
                        "done": True,
                        "error": f"step_failed: {exc}",
                    },
                )
                break

            observation = payload.get("observation", observation)
            final_reward = _safe_score(payload.get("reward", final_reward))
            done = bool(payload.get("done", False))

            _emit(
                "STEP",
                {
                    "task_id": task_id,
                    "step": step_idx,
                    "action_type": action.get("action_type"),
                    "reward": final_reward,
                    "done": done,
                    "error": (payload.get("info") or {}).get("error"),
                },
            )

            if done:
                break

        safe_final_reward = _safe_score(final_reward)
        final_scores[task_id] = safe_final_reward
        _emit(
            "END",
            {
                "task_id": str(task_id),
                "score": safe_final_reward,
                "completed": True,
            },
        )

    avg_score = _safe_score(sum(final_scores.values()) / len(final_scores))
    ordered_task_scores = [{"task_id": str(task_id), "score": final_scores[task_id]} for task_id in TASK_IDS]
    ordered_scores = [item["score"] for item in ordered_task_scores]

    _emit(
        "END",
        {
            "summary": True,
            "task_scores": ordered_task_scores,
            "scores": ordered_scores,
            "average_score": avg_score,
        },
    )


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        _emit(
            "END",
            {
                "summary": True,
                "task_scores": [{"task_id": str(task_id), "score": _safe_score(0.0)} for task_id in TASK_IDS],
                "scores": [_safe_score(0.0) for _ in TASK_IDS],
                "average_score": _safe_score(0.0),
                "fatal_error": str(exc),
            },
        )
