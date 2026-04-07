from __future__ import annotations

import json
import os
from typing import Any

import requests
from openai import OpenAI

MAX_STEPS = 8
TASK_IDS = [1, 2, 3]


def _emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), sort_keys=False)}")


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _build_client() -> OpenAI:
    api_base_url = _get_required_env("API_BASE_URL")
    api_key = os.getenv("HF_TOKEN", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing required environment variable: HF_TOKEN or OPENAI_API_KEY")
    return OpenAI(base_url=api_base_url, api_key=api_key)


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


def choose_action(client: OpenAI, model_name: str, observation: dict[str, Any]) -> dict[str, Any]:
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

    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(prompt)},
        ],
    )

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
        parsed["action_type"] = "execute_sql"
        parsed["query"] = observation.get("current_query", "SELECT 1")

    return {
        "action_type": parsed.get("action_type"),
        "query": parsed.get("query"),
        "table": parsed.get("table"),
        "column": parsed.get("column"),
    }


def run() -> None:
    model_name = _get_required_env("MODEL_NAME")
    env_base_url = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")

    client = _build_client()

    final_scores: dict[int, float] = {}

    for task_id in TASK_IDS:
        _emit(
            "START",
            {
                "task_id": task_id,
                "model": model_name,
                "max_steps": MAX_STEPS,
                "env_base_url": env_base_url,
            },
        )

        reset_resp = requests.post(f"{env_base_url}/reset", json={"task_id": task_id}, timeout=30)
        reset_resp.raise_for_status()
        observation = reset_resp.json()

        final_reward = 0.0

        for step_idx in range(1, MAX_STEPS + 1):
            action = choose_action(client, model_name, observation)
            step_resp = requests.post(f"{env_base_url}/step", json={"action": action}, timeout=45)
            step_resp.raise_for_status()

            payload = step_resp.json()
            observation = payload["observation"]
            final_reward = float(payload["reward"])

            _emit(
                "STEP",
                {
                    "task_id": task_id,
                    "step": step_idx,
                    "action_type": action.get("action_type"),
                    "reward": final_reward,
                    "done": bool(payload.get("done")),
                    "error": (payload.get("info") or {}).get("error"),
                },
            )

            if payload["done"]:
                break

        final_scores[task_id] = final_reward
        _emit(
            "END",
            {
                "task_id": task_id,
                "final_reward": final_reward,
                "completed": True,
            },
        )

    avg_score = sum(final_scores.values()) / len(final_scores)

    _emit(
        "END",
        {
            "summary": True,
            "scores": final_scores,
            "average_reward": round(avg_score, 6),
        },
    )


if __name__ == "__main__":
    run()
