from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping, Sequence

from openai import OpenAI

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import TrainingSupervisorState


OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY_GARRY_CHESS")
EXPENSIVE_MODEL = os.environ.get("TRAINING_SUPERVISOR_EXPENSIVE_MODEL", "gpt-5")
CHEAP_MODEL = os.environ.get("TRAINING_SUPERVISOR_CHEAP_MODEL", "gpt-4.1-mini")
TRACE_LIMIT = 25


def build_common_step_context(state: TrainingSupervisorState) -> dict[str, Any]:
    scratch = state.scratch
    return {
        "plan_view": state.plan_view.to_dict(),
        "registry_summary": state.registry_summary.to_dict(),
        "queue": [item.to_dict() for item in state.queue],
        "proposed_studies": [proposal.to_dict() for proposal in state.proposed_studies],
        "active_run": state.active_run.to_dict() if state.active_run else None,
        "latest_observation": state.latest_observation.to_dict() if state.latest_observation else None,
        "shared_step_context": dict(scratch.get("shared_step_context", {})),
        "prior_step_decisions": dict(scratch.get("step_decisions", {})),
        "llm_trace_tail": list(scratch.get("llm_trace", []))[-5:],
        "last_step": state.last_step,
        "last_action": state.last_action,
        "loop_iteration": state.loop_iteration,
    }


def call_step_llm(
    *,
    state: TrainingSupervisorState,
    step_name: str,
    system_goal: str,
    allowed_actions: Sequence[str],
    decision_schema_hint: Mapping[str, Any],
    step_context: Mapping[str, Any],
) -> dict[str, Any]:
    llm_calls = int(state.scratch.get("llm_calls", 0))
    model = EXPENSIVE_MODEL if llm_calls == 0 else CHEAP_MODEL

    allowed = list(allowed_actions)
    if "wait" not in allowed:
        allowed.append("wait")

    if not OPEN_AI_API_KEY:
        decision = {
            "action": "wait",
            "reason": "missing_openai_api_key",
            "reasoning_summary": "No OpenAI API key configured; defaulting to no-op.",
            "shared_context_updates": {},
        }
        return {
            "decision": decision,
            "model": model,
            "llm_calls_next": llm_calls + 1,
        }

    client = OpenAI(api_key=OPEN_AI_API_KEY)

    prompt = f"""
You are making a decision for ONE step in a training-only ML experiment supervisor.

Global goal:
{system_goal}

Current step:
{step_name}

Hard scope rules:
- This supervisor is ONLY for training jobs.
- Do NOT schedule metric-eval jobs.
- Do NOT schedule graph-generation jobs.
- You may reference downstream evaluation or plotting needs only to decide which TRAINING studies matter.

Allowed actions:
{json.dumps(allowed, indent=2)}

Return strict JSON only.
No markdown fences.
No prose outside JSON.

Every response MUST include:
- action
- reason
- reasoning_summary
- shared_context_updates

Decision schema:
{json.dumps(decision_schema_hint, indent=2)}

State for this step:
{json.dumps(step_context, indent=2)}

Note: v2 variant is now available.
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "text"},
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response was not a JSON object")
        decision = parsed
    except Exception as exc:
        decision = {
            "action": "wait",
            "reason": f"llm_error:{type(exc).__name__}",
            "reasoning_summary": "LLM call failed; defaulting to no-op.",
            "shared_context_updates": {},
        }

    if decision.get("action") not in set(allowed):
        decision = {
            "action": "wait",
            "reason": f"invalid_action:{decision.get('action')}",
            "reasoning_summary": "LLM returned an invalid action; defaulting to no-op.",
            "shared_context_updates": {},
        }

    if not isinstance(decision.get("shared_context_updates", {}), Mapping):
        decision["shared_context_updates"] = {}

    return {
        "decision": decision,
        "model": model,
        "llm_calls_next": llm_calls + 1,
    }


def record_step_decision(
    state: TrainingSupervisorState,
    *,
    step_name: str,
    llm_result: Mapping[str, Any],
    step_context_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    scratch = dict(state.scratch)

    llm_trace = list(scratch.get("llm_trace", []))
    step_decisions = dict(scratch.get("step_decisions", {}))
    shared_before = dict(scratch.get("shared_step_context", {}))

    decision = dict(llm_result.get("decision", {}))
    updates = decision.get("shared_context_updates", {})
    if not isinstance(updates, Mapping):
        updates = {}

    shared_after = deep_merge_dicts(shared_before, dict(updates))

    trace_entry = {
        "time": time.time(),
        "step_name": step_name,
        "model": llm_result.get("model"),
        "action": decision.get("action"),
        "reason": decision.get("reason"),
        "reasoning_summary": decision.get("reasoning_summary"),
        "shared_context_updates": dict(updates),
        "context_digest": {
            "active_run": state.active_run.study_name if state.active_run else None,
            "latest_status": state.latest_observation.status if state.latest_observation else None,
            "queue_size": len(state.queue),
            "proposed_count": len(state.proposed_studies),
            "step_context_keys": sorted(step_context_snapshot.keys()),
        },
    }

    llm_trace.append(trace_entry)
    llm_trace = llm_trace[-TRACE_LIMIT:]

    step_decisions[step_name] = decision
    scratch["llm_trace"] = llm_trace
    scratch["step_decisions"] = step_decisions
    scratch["shared_step_context"] = shared_after
    scratch["llm_calls"] = int(llm_result.get("llm_calls_next", 0))
    return scratch


def deep_merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge_dicts(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged
