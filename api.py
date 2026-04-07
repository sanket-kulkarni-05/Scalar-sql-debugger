from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env.actions import ActionModel, ObservationModel
from env.environment import SQLDebuggerEnvironment

app = FastAPI(title="SQL Query Debugger OpenEnv", version="1.0.0")
env = SQLDebuggerEnvironment()


class ResetRequest(BaseModel):
    task_id: int = Field(default=1, ge=1, le=3)


class StepRequest(BaseModel):
    action: ActionModel


class StepResponse(BaseModel):
    observation: ObservationModel
    reward: float
    done: bool
    info: dict[str, Any]


@app.get("/")
def root_endpoint() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "SQL Query Debugger OpenEnv",
        "docs": "/docs",
    }


@app.get("/health")
def health_endpoint() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset", response_model=ObservationModel)
def reset_endpoint(payload: ResetRequest | None = None) -> ObservationModel:
    try:
        task_id = payload.task_id if payload is not None else 1
        return env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}") from exc


@app.post("/step", response_model=StepResponse)
def step_endpoint(payload: StepRequest) -> StepResponse:
    try:
        observation, reward, done, info = env.step(payload.action)
        return StepResponse(observation=observation, reward=reward, done=done, info=info)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # This path should be rare because env.step has crash-safe fallback semantics.
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}") from exc


@app.get("/state")
def state_endpoint() -> dict[str, Any]:
    return env.state()
