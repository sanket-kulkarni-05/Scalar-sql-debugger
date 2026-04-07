from __future__ import annotations

import importlib
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from env.actions import ActionModel, ActionType, ObservationModel
from env.database import seed_database
from env.grader import grade_submission


MAX_STEPS = 8
VALID_ACTIONS = {item.value for item in ActionType}
STRICT_SCORE_EPSILON = 1e-6


def _strict_unit_interval(value: float) -> float:
    return max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, float(value)))


@dataclass
class EpisodeState:
    task_id: int
    step_count: int
    current_query: str
    last_result: list[Any] | None
    execution_plan: list[Any] | None
    done: bool
    indexes_added: list[str] = field(default_factory=list)


class SQLDebuggerEnvironment:
    def __init__(self, db_path: str | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        self.db_path = db_path or str(root / "env.db")
        self.conn: sqlite3.Connection | None = None
        self.episode_state: EpisodeState | None = None
        self.current_task: dict[str, Any] | None = None

    def _connect(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON;")
        return self.conn

    def _close_connection(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _load_task(self, task_id: int) -> dict[str, Any]:
        task_map = {
            1: "env.tasks.task_easy",
            2: "env.tasks.task_medium",
            3: "env.tasks.task_hard",
        }

        if task_id not in task_map:
            raise ValueError(f"Unknown task_id: {task_id}")

        module = importlib.import_module(task_map[task_id])
        task = dict(module.TASK)

        conn = self._connect()
        if not task.get("expected_rows"):
            rows = conn.execute(task["expected_query"]).fetchall()
            task["expected_rows"] = [tuple(row) for row in rows]

        if not task.get("baseline_cost"):
            baseline_plan = conn.execute(f"EXPLAIN QUERY PLAN {task['broken_query']}").fetchall()
            task["baseline_cost"] = len(baseline_plan)

        task["baseline_plan"] = conn.execute(f"EXPLAIN QUERY PLAN {task['broken_query']}").fetchall()
        return task

    def _schema_info(self) -> dict[str, Any]:
        conn = self._connect()
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        schema: dict[str, Any] = {"tables": {}}

        for row in tables:
            table_name = row["name"]
            col_rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            idx_rows = conn.execute(f"PRAGMA index_list({table_name})").fetchall()

            schema["tables"][table_name] = {
                "columns": [
                    {
                        "name": col["name"],
                        "type": col["type"],
                        "notnull": bool(col["notnull"]),
                        "pk": bool(col["pk"]),
                    }
                    for col in col_rows
                ],
                "indexes": [
                    {
                        "name": idx["name"],
                        "unique": bool(idx["unique"]),
                        "origin": idx["origin"],
                    }
                    for idx in idx_rows
                ],
            }

        return schema

    def _build_observation(self) -> ObservationModel:
        if self.episode_state is None or self.current_task is None:
            raise RuntimeError("Environment is not initialized. Call reset(task_id) first.")

        return ObservationModel(
            task_description=self.current_task["description"],
            schema_info=self._schema_info(),
            current_query=self.episode_state.current_query,
            last_result=self.episode_state.last_result,
            execution_plan=self.episode_state.execution_plan,
            step_count=self.episode_state.step_count,
            done=self.episode_state.done,
        )

    def reset(self, task_id: int) -> ObservationModel:
        self._close_connection()

        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        seed_database(self.db_path)
        self._connect()

        self.current_task = self._load_task(task_id)
        self.episode_state = EpisodeState(
            task_id=task_id,
            step_count=0,
            current_query=self.current_task["broken_query"],
            last_result=None,
            execution_plan=None,
            done=False,
            indexes_added=[],
        )

        return self._build_observation()

    def _invalid_action_result(self, message: str) -> tuple[ObservationModel, float, bool, dict[str, Any]]:
        observation = self._build_observation()
        return observation, _strict_unit_interval(0.0), False, {"error": message}

    def _validate_action(self, action: ActionModel) -> str | None:
        if action.action_type not in VALID_ACTIONS:
            return f"Invalid action_type: {action.action_type}"

        if action.action_type in {ActionType.EXECUTE_SQL.value, ActionType.EXPLAIN_PLAN.value, ActionType.REWRITE_QUERY.value, ActionType.SUBMIT_ANSWER.value}:
            if not action.query:
                return f"Action '{action.action_type}' requires field 'query'"

        if action.action_type == ActionType.ADD_INDEX.value:
            if not action.table or not action.column:
                return "Action 'add_index' requires fields 'table' and 'column'"

        return None

    def _execute_sql(self, query: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            rows = conn.execute(query).fetchall()
            serialized = [dict(row) for row in rows]
            self.episode_state.last_result = serialized
            self.episode_state.execution_plan = None
            return {"rows": len(serialized)}
        except sqlite3.Error as exc:
            self.episode_state.last_result = [{"error": str(exc)}]
            self.episode_state.execution_plan = None
            return {"error": str(exc)}

    def _explain_plan(self, query: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            plan_rows = conn.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
            serialized = [dict(row) for row in plan_rows]
            self.episode_state.execution_plan = serialized
            self.episode_state.last_result = None
            return {"plan_steps": len(serialized)}
        except sqlite3.Error as exc:
            self.episode_state.execution_plan = [{"error": str(exc)}]
            self.episode_state.last_result = None
            return {"error": str(exc)}

    def _add_index(self, table: str, column: str) -> dict[str, Any]:
        conn = self._connect()
        index_name = f"idx_{table}_{column}"
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({column})")
            conn.commit()
            if index_name not in self.episode_state.indexes_added:
                self.episode_state.indexes_added.append(index_name)
            return {"index": index_name}
        except sqlite3.Error as exc:
            return {"error": str(exc)}

    def _rewrite_query(self, query: str) -> dict[str, Any]:
        self.episode_state.current_query = query
        return {"rewritten": True}

    def _submit_answer(self, query: str) -> tuple[float, dict[str, Any]]:
        conn = self._connect()
        reward, info = grade_submission(
            conn=conn,
            sql=query,
            expected_rows=self.current_task["expected_rows"],
            baseline_plan=self.current_task.get("baseline_plan"),
            step_count=self.episode_state.step_count,
        )
        self.episode_state.done = True
        self.episode_state.last_result = [{"submitted": True}]
        return reward, info

    def step(self, action: ActionModel | dict[str, Any]) -> tuple[ObservationModel, float, bool, dict[str, Any]]:
        if self.episode_state is None or self.current_task is None:
            raise RuntimeError("Environment is not initialized. Call reset(task_id) first.")

        if self.episode_state.done:
            observation = self._build_observation()
            return observation, _strict_unit_interval(0.0), True, {"message": "Episode already finished"}

        if self.episode_state.step_count >= MAX_STEPS:
            self.episode_state.done = True
            observation = self._build_observation()
            return observation, _strict_unit_interval(0.0), True, {"message": "Max steps reached"}

        parsed = action if isinstance(action, ActionModel) else ActionModel(**action)

        validation_error = self._validate_action(parsed)
        if validation_error:
            return self._invalid_action_result(validation_error)

        info: dict[str, Any] = {}
        reward = _strict_unit_interval(0.0)

        try:
            if parsed.action_type == ActionType.EXECUTE_SQL.value:
                info = self._execute_sql(parsed.query or "")
            elif parsed.action_type == ActionType.EXPLAIN_PLAN.value:
                info = self._explain_plan(parsed.query or "")
            elif parsed.action_type == ActionType.ADD_INDEX.value:
                info = self._add_index(parsed.table or "", parsed.column or "")
            elif parsed.action_type == ActionType.REWRITE_QUERY.value:
                info = self._rewrite_query(parsed.query or "")
            elif parsed.action_type == ActionType.SUBMIT_ANSWER.value:
                reward, info = self._submit_answer(parsed.query or "")
            else:
                return self._invalid_action_result(f"Unsupported action: {parsed.action_type}")
        except Exception as exc:  # noqa: BLE001
            return self._invalid_action_result(f"Action execution failed: {exc}")

        self.episode_state.step_count += 1

        if self.episode_state.step_count >= MAX_STEPS and not self.episode_state.done:
            self.episode_state.done = True

        observation = self._build_observation()
        return observation, _strict_unit_interval(reward), self.episode_state.done, info

    def state(self) -> dict[str, Any]:
        if self.episode_state is None:
            return {"initialized": False}

        return {
            "initialized": True,
            "db_path": self.db_path,
            "task": {
                "id": self.current_task["id"],
                "name": self.current_task["name"],
                "difficulty": self.current_task["difficulty"],
            },
            "episode_state": asdict(self.episode_state),
        }
