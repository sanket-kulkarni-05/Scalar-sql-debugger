---
title: SQL Debugger
emoji: 🏢
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
---

# SQL Query Debugger OpenEnv

## Overview
SQL Query Debugger OpenEnv is a production-style environment for training and evaluating agents that debug and optimize SQL on relational systems. It models a realistic workflow used by analysts and backend engineers: inspect schema, run SQL, explain plans, add indexes, rewrite queries, and submit a final fix.

## Real-World Motivation
Teams regularly lose time on incorrect and slow SQL in reporting and transactional pipelines. This environment simulates that pressure with correctness and performance scoring, helping improve agent quality for practical SQL troubleshooting.

## Architecture
The environment separates state into two layers:
- Persistent state: SQLite database file `env.db`.
- Episode state: in-memory `EpisodeState` object (task id, step count, query/result/plan snapshots, completion flag, created indexes).

`reset(task_id)` fully wipes both layers by deleting `env.db`, reseeding deterministic data, and reinitializing episode state.

## Project Structure
- `env/database.py`: deterministic schema and data seeding.
- `env/environment.py`: reset/step/state lifecycle and action routing.
- `env/actions.py`: typed Pydantic action/observation/reward contracts.
- `env/grader.py`: deterministic grading and plan-based heuristic performance reward.
- `env/tasks/task_easy.py`: correctness-focused bug.
- `env/tasks/task_medium.py`: aggregation bug.
- `env/tasks/task_hard.py`: high-quality optimization benchmark (multi-table join + aggregation + date filter + order/limit).
- `api.py`: FastAPI server with `/reset`, `/step`, `/state`.
- `inference.py`: LLM runner loop over all tasks.
- `openenv.yaml`: OpenEnv metadata.

## Action Space
Supported `action_type` values:
- `execute_sql` with `query`
- `explain_plan` with `query`
- `add_index` with `table` and `column`
- `rewrite_query` with `query`
- `submit_answer` with `query`

Invalid action payloads do not crash the environment. They return unchanged observation, a tiny non-zero reward inside `(0, 1)`, `done=false`, and `info.error`.

## Observation Space
Each step returns:
- `task_description`
- `schema_info` (table names, columns, types, indexes from PRAGMA)
- `current_query`
- `last_result`
- `execution_plan`
- `step_count`
- `done`

## Tasks
1. Easy: fix a simple filter logic bug.
2. Medium: fix GROUP BY / aggregation semantics.
3. Hard: top 5 customers by total spending in last year, requiring both logic correction and performance optimization opportunities.

## Setup
```bash
cd sql-debugger-env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 7860
```

## Docker
```bash
cd sql-debugger-env
docker build -t sql-debugger-env .
docker run --rm -p 7860:7860 sql-debugger-env
```

## Hugging Face Spaces Deployment Notes
1. Push this directory to your Space repository.
2. Ensure `Dockerfile` is present.
3. Expose port `7860`.
4. Add environment variables for inference usage if needed:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
   - `OPENENV_BASE_URL`

## Inference Runner
The runner uses OpenAI-compatible client calls and then drives the environment API.
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="<token>"
export OPENENV_BASE_URL="http://localhost:7860"
python inference.py
```

## Baseline Results Format
`inference.py` prints:
- Per-task final reward
- Aggregate average reward

Because seeding and grading are deterministic, repeated runs with identical model outputs produce identical environment outcomes.
