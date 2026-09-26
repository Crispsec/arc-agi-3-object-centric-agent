"""
ReAct Planner — local LLM reasoning layer for ARC-AGI-3.

Uses Ollama (default: qwen3:4b) running locally — no external API calls.

Called every N steps or when the low-level planner is stuck.
Maintains a multi-turn conversation so the LLM builds understanding
across calls rather than reacting to each snapshot in isolation.

Output: a structured GoalHypothesis (parsed from the LLM's JSON block)
that the existing low-level Planner executes action-by-action.

Fallback: if Ollama is unreachable or the response is unparseable,
the last valid goal is reused (or the caller falls back to heuristic).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Dict, List, Optional

import ollama

from dataclasses import dataclass, field

from embedding_space import EmbeddingSpace, ObjectEntry
from goal_planner import GoalHypothesis, GoalType

log = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3:4b"
MAX_TOKENS = 600
HISTORY_TURNS = 6          # keep last N user+assistant pairs
CALL_EVERY_N_STEPS = 15
STREAM_TIMEOUT_SEC = 15    # abort streaming if no chunk arrives within this window


# ------------------------------------------------------------------
# Color names
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Action plan (open goal system)
# ------------------------------------------------------------------

@dataclass
class ActionPlan:
    """
    Open-ended plan produced by the LLM.

    goal        : natural language description of what to achieve
    actions     : ordered list of (action_name, data) to execute
    fallback_goal: a GoalHypothesis to hand to the heuristic planner
                   if the action queue runs out with no replan
    """
    goal: str
    actions: list = field(default_factory=list)   # list of {"action": str, "x": int, "y": int}
    fallback_goal: Optional[GoalHypothesis] = None


# ------------------------------------------------------------------
# Color names
# ------------------------------------------------------------------

_COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "magenta", 7: "orange", 8: "cyan", 9: "maroon",
    10: "purple", 11: "teal", 12: "brown", 13: "lime", 14: "navy", 15: "white",
}


# ------------------------------------------------------------------
# State summariser
# ------------------------------------------------------------------

def _describe_object(oid: str, entry: ObjectEntry) -> str:
    color = _COLOR_NAMES.get(entry.properties.get("color", 0), "unknown")
    size = entry.properties.get("size", 0)
    size_s = "tiny" if size < 4 else "small" if size < 16 else "medium" if size < 64 else "large"
    cr, cc = entry.properties.get("centroid", (0, 0))
    concept = entry.visual_concept or "unknown"
    return (
        f"  [{oid[:6]}] {size_s} {color} @ ({int(cr)},{int(cc)}) "
        f"concept='{concept}' conf={entry.confidence:.2f} interest={entry.interestingness:.1f}"
    )


def build_state_summary(
    space: EmbeddingSpace,
    recent_actions: List[str],
    step: int,
    level: int,
) -> str:
    lines = [f"=== Step {step}, Level {level} ==="]

    entries = sorted(
        space.objects.items(),
        key=lambda x: -(x[1].interestingness + x[1].confidence),
    )[:20]

    known = [(oid, e) for oid, e in entries if e.confidence >= 0.3]
    unknown = [(oid, e) for oid, e in entries if e.confidence < 0.3]

    lines.append(f"\nKnown objects ({len(known)}):")
    for oid, e in known:
        lines.append(_describe_object(oid, e))

    if unknown:
        lines.append(f"\nUnknown objects (top {min(5, len(unknown))} of {len(unknown)}):")
        for oid, e in unknown[:5]:
            lines.append(_describe_object(oid, e))

    if recent_actions:
        lines.append("\nRecent actions (newest last):")
        for a in recent_actions[-10:]:
            lines.append(f"  {a}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Response parser
# ------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that qwen3 may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_action_plan(response_text: str, space: EmbeddingSpace) -> Optional[ActionPlan]:
    """
    Extract an ActionPlan from the LLM's JSON block.

    Expected format:
    ```json
    {
      "goal": "natural language goal",
      "actions": [{"action": "CLICK", "x": 14, "y": 36}, {"action": "RIGHT"}],
      "reasoning": "one sentence"
    }
    ```

    Also accepts the old GoalHypothesis format as a fallback.
    """
    text = _strip_thinking(response_text)

    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"\{[^{}]*\"(?:goal|goal_type)\"[^{}]*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        raw = match.group(1) if match.lastindex else match.group(0)
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    reasoning = data.get("reasoning", "")
    if reasoning:
        log.info(f"[ReAct] {reasoning}")

    # New open-goal format
    if "actions" in data:
        goal_text = data.get("goal", "unknown goal")
        actions = data.get("actions", [])
        # Build a fallback GoalHypothesis from the first CLICK target if available
        fallback = _actions_to_fallback(actions, space)
        return ActionPlan(goal=goal_text, actions=actions, fallback_goal=fallback)

    # Legacy format: {"goal_type": ..., "object_a": ..., "object_b": ...}
    raw_type = data.get("goal_type") or data.get("goal", "bring")
    if isinstance(raw_type, str) and raw_type.lower() in ("bring", "avoid", "reach"):
        goal_type = {"bring": GoalType.BRING, "avoid": GoalType.AVOID, "reach": GoalType.REACH}[raw_type.lower()]
        oid_a = _resolve_ref(str(data.get("object_a", "")), space)
        oid_b = _resolve_ref(str(data.get("object_b", "")), space) if data.get("object_b") else None
        if oid_a is None:
            return None
        fallback = GoalHypothesis(goal_type=goal_type, object_a=oid_a, object_b=oid_b, probability=0.8)
        return ActionPlan(goal=f"{raw_type} {oid_a[:6]}", actions=[], fallback_goal=fallback)

    return None


def _actions_to_fallback(actions: list, space: EmbeddingSpace) -> Optional[GoalHypothesis]:
    """Build a rough GoalHypothesis from the first CLICK in the action list."""
    for act in actions:
        if act.get("action") == "CLICK":
            x, y = act.get("x", 0), act.get("y", 0)
            # Find nearest object to (y, x) = (row, col)
            best_oid, best_dist = None, float("inf")
            for oid, entry in space.objects.items():
                cr, cc = entry.properties.get("centroid", (0, 0))
                d = ((cr - y) ** 2 + (cc - x) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, best_oid = d, oid
            if best_oid:
                return GoalHypothesis(goal_type=GoalType.REACH, object_a=best_oid, probability=0.7)
    return None


def _resolve_ref(ref: str, space: EmbeddingSpace) -> Optional[str]:
    """Map an LLM object reference (ID prefix or color name) to a space object ID."""
    if not ref:
        return None

    # ID prefix match
    for oid in space.object_ids():
        if oid.startswith(ref) or ref.startswith(oid[:6]):
            return oid

    # Color name match — pick highest interestingness
    ref_lower = ref.lower()
    color_id = next((cid for cid, name in _COLOR_NAMES.items() if name in ref_lower), None)
    if color_id is not None:
        candidates = [
            (oid, e) for oid, e in space.objects.items()
            if e.properties.get("color") == color_id
        ]
        if candidates:
            candidates.sort(key=lambda x: -x[1].interestingness)
            return candidates[0][0]

    return None


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = """You are the reasoning engine for an AI agent playing unknown puzzle games on a 64x64 grid.

The agent executor supports:
  CLICK x y        — click a specific grid cell
  UP / DOWN / LEFT / RIGHT  — directional moves (no coordinates needed)

Your job:
1. Reason about what the puzzle mechanic is (step by step).
2. Output a natural language goal AND a concrete action plan.

You receive:
- Known objects: [ID] size color @ (row,col) concept='...' conf=N.NN interest=N.N
- Recent actions with outcome (→ N object(s) changed)

Guidelines:
- High interestingness means the object reacted to something — it matters.
- Objects with concept 'enemy or hazard' are dangerous — avoid clicking them.
- If actions keep having no effect, try a different object or direction.
- Batch 3-5 actions per plan. The agent replans automatically if something changes.
- Prefer CLICK (row, col) for objects. Use directional keys only if no clear target.

Output format — one JSON block at the end of your reasoning:
```json
{
  "goal": "one sentence describing what you are trying to achieve",
  "actions": [
    {"action": "CLICK", "x": 14, "y": 36},
    {"action": "RIGHT"},
    {"action": "CLICK", "x": 28, "y": 10}
  ],
  "reasoning": "one sentence why this plan makes sense"
}
```

action values: CLICK (requires x=col, y=row), UP, DOWN, LEFT, RIGHT
x = column (0-63), y = row (0-63)
Object positions are shown as (row, col). For CLICK: x=col, y=row.
Always output exactly one JSON block."""


# ------------------------------------------------------------------
# ReAct Planner
# ------------------------------------------------------------------

class ReActPlanner:
    """
    Local LLM reasoning layer using Ollama.

    Parameters
    ----------
    model : str
        Ollama model tag to use.
    call_every : int
        Minimum steps between LLM calls.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        call_every: int = CALL_EVERY_N_STEPS,
    ):
        self.model = model
        self.call_every = call_every
        self._history: List[Dict[str, str]] = []
        self._last_plan: Optional[ActionPlan] = None
        self._last_call_step: int = -999
        self._action_log: List[str] = []
        self._available: Optional[bool] = None   # lazily checked

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                models = [m.model for m in ollama.list().models]
                self._available = any(self.model in m for m in models)
                if not self._available:
                    log.warning(
                        f"[ReAct] Model '{self.model}' not found in Ollama "
                        f"(available: {models}). Falling back to heuristic planner."
                    )
                else:
                    log.info(f"[ReAct] Using local model: {self.model}")
            except Exception as e:
                log.warning(f"[ReAct] Ollama not reachable ({e}). Falling back to heuristic planner.")
                self._available = False
        return self._available

    def log_action(self, action_name: str, data: dict, changed_objects: int) -> None:
        entry = action_name
        if data:
            entry += f"({data})"
        entry += f" → {changed_objects} object(s) changed" if changed_objects > 0 else " → no change"
        self._action_log.append(entry)
        if len(self._action_log) > 30:
            self._action_log = self._action_log[-30:]

    def should_call(self, step: int, stuck_count: int, stuck_threshold: int) -> bool:
        if not self.available:
            return False
        periodic = (step - self._last_call_step) >= self.call_every
        stuck = stuck_count >= stuck_threshold
        return periodic or stuck

    def reason(
        self,
        space: EmbeddingSpace,
        step: int,
        level: int,
    ) -> Optional[ActionPlan]:
        """
        Query the local LLM and return an ActionPlan.
        Returns None on failure — caller uses heuristic fallback.
        """
        if not self.available:
            return None

        self._last_call_step = step
        summary = build_state_summary(space, self._action_log, step, level)
        self._history.append({"role": "user", "content": summary})

        # Trim to last N turns
        if len(self._history) > HISTORY_TURNS * 2:
            self._history = self._history[-(HISTORY_TURNS * 2):]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        try:
            # Stream response — qwen3 exhausts num_predict on thinking if not streamed.
            # Use a per-chunk timeout: if no chunk arrives within STREAM_TIMEOUT_SEC,
            # a daemon thread sets the stop_event and we break out of the loop.
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": 0.3},
            )
            content_parts = []
            timed_out = False
            stop_event = threading.Event()

            def _watchdog():
                if not stop_event.wait(timeout=STREAM_TIMEOUT_SEC):
                    nonlocal timed_out
                    timed_out = True

            watchdog = threading.Thread(target=_watchdog, daemon=True)
            watchdog.start()

            for chunk in stream:
                if timed_out:
                    log.warning(f"[ReAct] Stream timeout after {STREAM_TIMEOUT_SEC}s — aborting")
                    break
                # Reset watchdog on each received chunk
                stop_event.set()
                stop_event = threading.Event()
                watchdog = threading.Thread(target=_watchdog, daemon=True)
                watchdog.start()
                if chunk.message.content:
                    content_parts.append(chunk.message.content)

            stop_event.set()  # signal last watchdog to exit
            text = "".join(content_parts)

            if timed_out and not text:
                log.warning("[ReAct] Timed out with no content — using last valid goal")
                return self._last_goal
            self._history.append({"role": "assistant", "content": text})

            plan = _parse_action_plan(text, space)
            if plan is not None:
                self._last_plan = plan
                log.info(
                    f"[ReAct] step={step} → goal='{plan.goal}' "
                    f"actions={len(plan.actions)}"
                )
            else:
                log.warning(f"[ReAct] Could not parse plan. Response: {_strip_thinking(text)[:200]}")

            return plan

        except Exception as e:
            log.warning(f"[ReAct] Inference failed: {e}")
            return self._last_plan

    def reset(self) -> None:
        """Call at the start of each level — clears conversation history."""
        self._history = []
        self._last_plan = None
        self._last_call_step = -999
        self._action_log = []


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    planner = ReActPlanner()
    if not planner.available:
        print(f"Ollama model '{planner.model}' not available. Run: ollama pull {planner.model}")
        sys.exit(1)

    from embedding_space import EmbeddingSpace, ObjectEntry

    space = EmbeddingSpace(slot_dim=64, meaning_dim=32)
    space._step = 10

    def _fake(oid, color, row, col, size, concept, interest, conf):
        e = ObjectEntry(
            object_id=oid,
            slot=torch.zeros(64),
            properties={"color": color, "centroid": (row, col),
                        "bbox": (row-2, col-2, row+2, col+2), "size": size, "shape_hash": 0},
            meaning_vector=torch.randn(32) * 0.5,
            confidence=conf,
        )
        e.visual_concept = concept
        e.interestingness = interest
        e.last_seen = 10
        return e

    space.objects["aabbcc001122"] = _fake("aabbcc001122", 4, 10, 15, 25, "key object", 1.4, 0.8)
    space.objects["ddeeff334455"] = _fake("ddeeff334455", 1, 40, 40, 80, "door or gate", 0.8, 0.7)
    space.objects["112233aabbcc"] = _fake("112233aabbcc", 2, 30, 30,  9, "enemy or hazard", 0.5, 0.6)

    planner.log_action("CLICK", {"x": 15, "y": 10}, changed_objects=2)
    planner.log_action("CLICK", {"x": 15, "y": 10}, changed_objects=0)
    planner.log_action("CLICK", {"x": 15, "y": 10}, changed_objects=0)

    print("\n--- Reasoning call ---")
    goal = planner.reason(space, step=15, level=1)
    print(f"Goal: {goal}")
    print("\nSmoke test complete.")
