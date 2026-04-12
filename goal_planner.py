"""
Phase 5: Goal Inference + Planner

GoalInferenceModule
-------------------
Maintains a ranked list of goal hypotheses. After each step it looks at:
  - The win/lose signal (obs.state)
  - The world summary vector (from RelationalTransformer)
  - The interaction history

Three hypothesis types (from the architecture plan):
  1. REACH     — "Move object X to position P"
  2. AVOID     — "Keep objects X and Y apart"
  3. BRING     — "Bring object X near object Y"

Hypothesis scoring (heuristic, no training required):
  - WIN signal: strongly boost whatever configuration was true just before win
  - GAME_OVER: penalise whatever was true just before loss
  - Pattern matching: if the relation graph shows strong R[i,j], hypothesise
    a BRING/AVOID relationship between those objects

Planner
-------
Given the best goal hypothesis and the relation graph, uses BFS over a
discrete object graph to find the next action.

BFS node = object configuration (which objects are near which)
Edge    = action that changes the configuration
Heuristic = distance from current config to goal config

For now, the "action that achieves X" is approximated by:
  - REACH goals: move the player/agent object toward target position
  - BRING goals: move the object toward the target object
  - AVOID goals: move the object away from the other object

This is a structural placeholder — the planner's edge model will be
upgraded once the relational transformer is trained (Phase 7+).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from arcengine import GameAction, GameState

from embedding_space import EmbeddingSpace, ObjectEntry
from relational_transformer import WorldState


# ------------------------------------------------------------------
# Hypothesis types
# ------------------------------------------------------------------

class GoalType(Enum):
    REACH = "reach"      # move object X to position P
    AVOID = "avoid"      # keep X and Y apart
    BRING = "bring"      # bring X near Y
    UNKNOWN = "unknown"  # no hypothesis yet


@dataclass
class GoalHypothesis:
    goal_type: GoalType
    object_a: str                           # primary object ID
    object_b: Optional[str] = None          # secondary (for BRING/AVOID)
    target_position: Optional[Tuple[float, float]] = None  # for REACH
    probability: float = 0.1
    evidence_count: int = 0

    def __str__(self) -> str:
        if self.goal_type == GoalType.REACH:
            return f"REACH({self.object_a[:6]} → {self.target_position}) p={self.probability:.2f}"
        if self.goal_type == GoalType.BRING:
            return f"BRING({self.object_a[:6]}, {self.object_b[:6]}) p={self.probability:.2f}"
        if self.goal_type == GoalType.AVOID:
            return f"AVOID({self.object_a[:6]}, {self.object_b[:6]}) p={self.probability:.2f}"
        return f"UNKNOWN p={self.probability:.2f}"


# ------------------------------------------------------------------
# Goal Inference Module
# ------------------------------------------------------------------

PROXIMITY_THRESHOLD = 8.0    # pixels — "near" definition
RELATION_STRONG_THRESHOLD = 0.5  # relation vector norm above this = strong link

class GoalInferenceModule:
    """
    Maintains and updates goal hypotheses.

    Parameters
    ----------
    max_hypotheses : int
        Maximum number of hypotheses to keep alive.
    win_boost : float
        Probability multiplier applied to consistent hypotheses on WIN.
    lose_penalty : float
        Probability multiplier on hypotheses active when GAME_OVER fires.
    """

    def __init__(
        self,
        max_hypotheses: int = 10,
        win_boost: float = 5.0,
        lose_penalty: float = 0.2,
    ):
        self.max_hypotheses: List[GoalHypothesis] = []
        self.max_h = max_hypotheses
        self.win_boost = win_boost
        self.lose_penalty = lose_penalty
        self._prev_state_snapshot: Optional[dict] = None  # snapshot before last action

    # ------------------------------------------------------------------

    def update(
        self,
        space: EmbeddingSpace,
        world_state: WorldState,
        game_state: GameState,
    ) -> None:
        """
        Update hypotheses from the current world state and game signal.
        """
        # Snapshot current object positions for future correlation
        snapshot = {
            oid: e.properties["centroid"]
            for oid, e in space.objects.items()
        }

        if game_state == GameState.WIN:
            self._on_win(snapshot)
        elif game_state == GameState.GAME_OVER:
            self._on_lose()

        # Generate new hypotheses from relation graph
        self._generate_from_relations(world_state, space)

        # Prune low-probability and duplicate hypotheses
        self._prune()

        self._prev_state_snapshot = snapshot

    # ------------------------------------------------------------------

    def _on_win(self, snapshot: dict) -> None:
        """Boost hypotheses that are consistent with the winning configuration."""
        for h in self.max_hypotheses:
            if h.goal_type == GoalType.BRING and h.object_b:
                pos_a = snapshot.get(h.object_a)
                pos_b = snapshot.get(h.object_b)
                if pos_a and pos_b:
                    dist = _distance(pos_a, pos_b)
                    if dist < PROXIMITY_THRESHOLD:
                        h.probability *= self.win_boost
                        h.evidence_count += 1
            elif h.goal_type == GoalType.REACH and h.target_position:
                pos_a = snapshot.get(h.object_a)
                if pos_a:
                    dist = _distance(pos_a, h.target_position)
                    if dist < PROXIMITY_THRESHOLD:
                        h.probability *= self.win_boost
                        h.evidence_count += 1

    def _on_lose(self) -> None:
        for h in self.max_hypotheses:
            h.probability *= self.lose_penalty

    def _generate_from_relations(
        self, world_state: WorldState, space: EmbeddingSpace
    ) -> None:
        """
        Look at the relation graph: strong pairwise relations suggest
        BRING or AVOID hypotheses.
        """
        oids = world_state.object_ids
        N = len(oids)
        if N < 2:
            return

        rg = world_state.relation_graph  # [N, N, relation_dim]

        # Find top-K strongest pairwise relations
        norms = rg.norm(dim=-1)  # [N, N]

        # Suppress self-relations
        for i in range(N):
            norms[i, i] = 0.0

        flat = norms.view(-1)
        top_k = min(5, flat.numel())
        top_indices = flat.topk(top_k).indices

        existing_pairs = {
            (h.object_a, h.object_b)
            for h in self.max_hypotheses
            if h.object_b is not None
        }

        for idx in top_indices:
            i = int(idx) // N
            j = int(idx) % N
            if norms[i, j].item() < RELATION_STRONG_THRESHOLD:
                continue
            oid_a = oids[i]
            oid_b = oids[j]

            if (oid_a, oid_b) in existing_pairs:
                continue

            # Default: hypothesise BRING (positive reward for proximity)
            h = GoalHypothesis(
                goal_type=GoalType.BRING,
                object_a=oid_a,
                object_b=oid_b,
                probability=0.15,
            )
            self.max_hypotheses.append(h)
            existing_pairs.add((oid_a, oid_b))

    def _prune(self) -> None:
        # Normalise probabilities
        total = sum(h.probability for h in self.max_hypotheses)
        if total > 0:
            for h in self.max_hypotheses:
                h.probability /= total

        # Keep top-K
        self.max_hypotheses.sort(key=lambda h: -h.probability)
        self.max_hypotheses = self.max_hypotheses[: self.max_h]

    # ------------------------------------------------------------------

    def best_goal(self) -> Optional[GoalHypothesis]:
        if not self.max_hypotheses:
            return None
        return self.max_hypotheses[0]

    def summarise(self) -> None:
        print(f"Goal hypotheses ({len(self.max_hypotheses)}):")
        for h in self.max_hypotheses[:5]:
            print(f"  {h}")


# ------------------------------------------------------------------
# Planner
# ------------------------------------------------------------------

# Map keyboard action to (dr, dc) direction
_ACTION_DIRECTION: Dict[GameAction, Tuple[int, int]] = {
    GameAction.ACTION1: (-1, 0),  # up
    GameAction.ACTION2: ( 1, 0),  # down
    GameAction.ACTION3: ( 0,-1),  # left
    GameAction.ACTION4: ( 0, 1),  # right
}


class Planner:
    """
    Selects an action that makes progress toward the best goal hypothesis.

    Uses a greedy geometric strategy rather than full BFS for now:
      - BRING: move the movable object toward the target object
      - AVOID: move the movable object away from the threat
      - REACH: move toward the target position

    "Movable object" heuristic: the smallest object (probably the player/agent).
    """

    def select_action(
        self,
        goal: GoalHypothesis,
        space: EmbeddingSpace,
        action_space: List[GameAction],
    ) -> Tuple[GameAction, dict]:
        """
        Returns (action, data_dict).
        """
        if goal is None or goal.goal_type == GoalType.UNKNOWN:
            return random.choice(action_space), {}

        entry_a = space.get_entry(goal.object_a)
        if entry_a is None:
            return random.choice(action_space), {}

        pos_a = entry_a.properties["centroid"]

        # Determine target position
        if goal.goal_type in (GoalType.BRING,) and goal.object_b:
            entry_b = space.get_entry(goal.object_b)
            if entry_b is None:
                return random.choice(action_space), {}
            target_pos = entry_b.properties["centroid"]
            direction = _toward(pos_a, target_pos)
        elif goal.goal_type == GoalType.AVOID and goal.object_b:
            entry_b = space.get_entry(goal.object_b)
            if entry_b is None:
                return random.choice(action_space), {}
            target_pos = entry_b.properties["centroid"]
            direction = _away(pos_a, target_pos)
        elif goal.goal_type == GoalType.REACH and goal.target_position:
            direction = _toward(pos_a, goal.target_position)
        else:
            return random.choice(action_space), {}

        # Try to find a click action first
        click_actions = [a for a in action_space if a.is_complex()]
        if click_actions:
            return click_actions[0], {
                "x": int(round(target_pos[1])),
                "y": int(round(target_pos[0])),
            }

        # Keyboard: pick action matching direction
        best_action = _direction_to_action(direction, action_space)
        return best_action, {}


def _toward(pos_a: Tuple, pos_b: Tuple) -> Tuple[float, float]:
    dr = pos_b[0] - pos_a[0]
    dc = pos_b[1] - pos_a[1]
    return (dr, dc)


def _away(pos_a: Tuple, pos_b: Tuple) -> Tuple[float, float]:
    dr, dc = _toward(pos_a, pos_b)
    return (-dr, -dc)


def _direction_to_action(
    direction: Tuple[float, float],
    action_space: List[GameAction],
) -> GameAction:
    dr, dc = direction
    if abs(dr) >= abs(dc):
        preferred = GameAction.ACTION2 if dr > 0 else GameAction.ACTION1
    else:
        preferred = GameAction.ACTION4 if dc > 0 else GameAction.ACTION3

    if preferred in action_space:
        return preferred
    return random.choice(action_space)


def _distance(a: Tuple, b: Tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import arc_agi
    from object_extractor import ObjectExtractor
    from curiosity_explorer import CuriosityExplorer, MEANING_DIM
    from relational_transformer import RelationalTransformer, InteractionMemory, InteractionRecord, EFFECT_DIM

    SLOT_DIM = 64
    INPUT_DIM = SLOT_DIM + MEANING_DIM  # 96

    arc = arc_agi.Arcade()
    env = arc.make("tr87-cd924810", render_mode=None)
    env.include_frame_data = True

    extractor = ObjectExtractor(slot_dim=SLOT_DIM, min_size=2)
    space = EmbeddingSpace(slot_dim=SLOT_DIM, meaning_dim=MEANING_DIM)
    explorer = CuriosityExplorer(action_space=env.action_space)
    memory = InteractionMemory(maxlen=200, input_dim=INPUT_DIM)
    rt = RelationalTransformer(input_dim=INPUT_DIM)
    goal_inf = GoalInferenceModule()
    planner = Planner()

    obs = env.reset()
    before_blobs = extractor(obs.frame[0])
    space.update(before_blobs)

    print("Running 30 steps — curiosity then planner...")
    for step in range(30):
        unknown = space.get_unknown_objects()

        if unknown:
            action, data, target_id = explorer.select_action(space)
            source = "explorer"
        else:
            # All objects known — use relational transformer + planner
            mat = space.to_matrix()
            oids = space.object_ids()
            mem_tensor = memory.to_tensor()
            with torch.no_grad():
                world_state = rt(mat, oids, mem_tensor)
            goal_inf.update(space, world_state, obs.state)
            goal = goal_inf.best_goal()
            action, data = planner.select_action(goal, space, env.action_space)
            target_id = None
            source = f"planner({goal})"

        if action is None:
            action = random.choice(env.action_space)
            data = {}

        obs = env.step(action, data=data or None)
        after_blobs = extractor(obs.frame[0])
        space.update(after_blobs)

        if target_id is not None:
            delta = explorer.record_interaction(target_id, before_blobs, after_blobs, space)
            target_entry = space.get_entry(target_id)
            if target_entry is not None:
                other_ids = [oid for oid in space.object_ids() if oid != target_id]
                if other_ids:
                    other_entry = space.get_entry(other_ids[0])
                    if other_entry is not None:
                        memory.record(InteractionRecord(
                            obj_a_embed=torch.cat([target_entry.slot, target_entry.meaning_vector]),
                            action_id=action.value - 1,
                            obj_b_embed=torch.cat([other_entry.slot, other_entry.meaning_vector]),
                            effect_embed=delta[:EFFECT_DIM],
                        ))

        state_str = f"[{obs.state.name}]" if obs.state != GameState.NOT_FINISHED else ""
        print(f"  Step {step+1:2d}: {source:12s} unknown={len(unknown):2d} "
              f"levels={obs.levels_completed}/{obs.win_levels} {state_str}")

        if obs.state in (GameState.WIN, GameState.GAME_OVER):
            break

        before_blobs = after_blobs

    print("\nFinal goal hypotheses:")
    goal_inf.summarise()
    print("\nPhase 5 complete.")
