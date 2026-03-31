from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    EXECUTE_SQL = "execute_sql"
    EXPLAIN_PLAN = "explain_plan"
    ADD_INDEX = "add_index"
    REWRITE_QUERY = "rewrite_query"
    SUBMIT_ANSWER = "submit_answer"


class ActionModel(BaseModel):
    # Keep fields optional so step() can perform semantic validation and fallback safely.
    action_type: Optional[str] = None
    query: Optional[str] = None
    table: Optional[str] = None
    column: Optional[str] = None


class ObservationModel(BaseModel):
    task_description: str
    schema_info: dict[str, Any]
    current_query: str
    last_result: Optional[list[Any]] = None
    execution_plan: Optional[list[Any]] = None
    step_count: int
    done: bool


class RewardModel(BaseModel):
    value: float = Field(ge=0.0, le=1.0)


class StepResultModel(BaseModel):
    observation: ObservationModel
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")
