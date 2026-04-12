"""
Phase 6: ObjectCentricAgent — Full Integration

Wires all five components into a single agent class that plays a complete
ARC-AGI-3 game end-to-end.

Full loop (one timestep):
  1. Receive grid from environment
  2. ObjectExtractor → N object blobs
  3. EmbeddingSpace.update() — new objects get zero meaning
  4. IF any objects have confidence < threshold:
       CuriosityExplorer generates exploratory action → step 8
  5. ELSE:
       RelationalTransformer reads all objects → relation graph + world summary
  6. GoalInferenceModule updates hypothesis set
  7. Planner selects action toward best goal
  8. Execute action → receive new grid
  9. Compute delta → update meaning vectors of affected objects
 10. Record interaction in memory
 11. Log action count (efficiency metric)

Cross-level transfer:
  When a level ends, the embedding space's object meanings and the relation
  graph patterns are serialised. At the next level they serve as priors:
  objects recognised by color+shape start with their earned meaning vector
  rather than zero.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from arcengine import GameAction, GameState

from object_extractor import ObjectExtractor, ObjectBlob
from embedding_space import EmbeddingSpace
from curiosity_explorer import CuriosityExplorer, MEANING_DIM, compute_delta
from relational_transformer import RelationalTransformer, InteractionMemory, InteractionRecord, EFFECT_DIM
from goal_planner import GoalInferenceModule, Planner, GoalHypothesis


log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Per-level stats
# ------------------------------------------------------------------

@dataclass
class LevelStats:
    level: int
    actions_taken: int = 0
    objects_seen: int = 0
    objects_known: int = 0
    final_state: str = ""
    exploration_steps: int = 0
    planning_steps: int = 0


# ------------------------------------------------------------------
# Cross-level prior
# ------------------------------------------------------------------

@dataclass
class ObjectPrior:
    """
    Carried from one level to the next.
    Keyed by (color, shape_hash) so it matches across levels
    even if centroid differs.
    """
    color: int
    shape_hash: int
    meaning_vector: torch.Tensor
    confidence: float


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

SLOT_DIM = 64
INPUT_DIM = SLOT_DIM + MEANING_DIM  # 96


class ObjectCentricAgent:
    """
    Full object-centric agent for ARC-AGI-3.

    Parameters
    ----------
    action_space : List[GameAction]
    confidence_threshold : float
        Below this, objects are explored before planning begins.
    min_object_size : int
        Minimum pixel count to count as an object.
    verbose : bool
        Print step-by-step logs.
    """

    def __init__(
        self,
        action_space: List[GameAction],
        confidence_threshold: float = 0.3,
        min_object_size: int = 2,
        verbose: bool = True,
    ):
        self.action_space = action_space
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

        # Components
        self.extractor = ObjectExtractor(slot_dim=SLOT_DIM, min_size=min_object_size)
        self.space = EmbeddingSpace(slot_dim=SLOT_DIM, meaning_dim=MEANING_DIM)
        self.explorer = CuriosityExplorer(
            action_space=action_space,
            confidence_threshold=confidence_threshold,
        )
        self.memory = InteractionMemory(maxlen=200, input_dim=INPUT_DIM)
        self.rt = RelationalTransformer(input_dim=INPUT_DIM)
        self.goal_inf = GoalInferenceModule()
        self.planner = Planner()

        # State
        self._prev_blobs: List[ObjectBlob] = []
        self._prev_action: Optional[GameAction] = None
        self._prev_target_id: Optional[str] = None
        self._action_count: int = 0
        self._level: int = 0
        self._level_stats: List[LevelStats] = []
        self._current_stats: LevelStats = LevelStats(level=0)

        # Cross-level priors: (color, shape_hash) → ObjectPrior
        self._priors: Dict[Tuple[int, int], ObjectPrior] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, grid: np.ndarray) -> None:
        """Call at the start of each level with the initial grid."""
        self._level += 1
        self._action_count = 0
        self._current_stats = LevelStats(level=self._level)

        # Reset per-level state
        self.space = EmbeddingSpace(slot_dim=SLOT_DIM, meaning_dim=MEANING_DIM)
        self.memory.clear()
        self.goal_inf = GoalInferenceModule()

        # Extract initial objects
        blobs = self.extractor(grid)
        self.space.update(blobs)
        self._apply_priors()
        self._prev_blobs = blobs

        if self.verbose:
            log.info(
                f"Level {self._level} started — "
                f"{len(blobs)} objects, "
                f"{len(self.space.get_unknown_objects())} unknown"
            )

    def step(self, grid: np.ndarray, game_state: GameState) -> Tuple[GameAction, dict]:
        """
        Process one frame and return the next (action, data).

        Parameters
        ----------
        grid : np.ndarray (H, W) int8
        game_state : GameState
        """
        self._action_count += 1
        self._current_stats.actions_taken += 1

        # 1-2. Extract objects
        blobs = self.extractor(grid)

        # 3. Update embedding space; get remapping so we can follow moved objects
        report = self.space.update(blobs)
        self._current_stats.objects_seen = len(self.space.objects)

        # Keep _prev_target_id current — if the tracked object moved and got
        # a new hash-ID, update our reference before computing the delta.
        old_to_new = report["old_to_new"]
        if self._prev_target_id is not None:
            self._prev_target_id = old_to_new.get(self._prev_target_id, self._prev_target_id)

        # Keep explorer's sticky target fresh too
        self.explorer.notify_id_remap(old_to_new)

        # 9-10. If we had a previous action, compute delta and record
        if self._prev_target_id is not None and self._prev_action is not None:
            self._record_interaction(blobs)

        unknown = self.space.get_unknown_objects(self.confidence_threshold)

        # 4. Exploration branch
        if unknown:
            self._current_stats.exploration_steps += 1
            action, data, target_id = self.explorer.select_action(self.space)
            if action is None:
                action, data = random.choice(self.action_space), {}
                target_id = None
            source = "explore"
        else:
            # 5-7. Planning branch
            self._current_stats.planning_steps += 1
            mat = self.space.to_matrix()
            if mat is not None:
                oids = self.space.object_ids()
                mem_tensor = self.memory.to_tensor()
                with torch.no_grad():
                    world_state = self.rt(mat, oids, mem_tensor)
                self.goal_inf.update(self.space, world_state, game_state)
                goal = self.goal_inf.best_goal()
                action, data = self.planner.select_action(goal, self.space, self.action_space)
            else:
                action, data = random.choice(self.action_space), {}
            target_id = None
            goal = self.goal_inf.best_goal()
            source = f"plan({goal.goal_type.value if goal else '?'})"

        self._prev_blobs = blobs
        self._prev_action = action
        self._prev_target_id = target_id

        if self.verbose:
            log.debug(
                f"Step {self._action_count:3d} [{source}] "
                f"unknown={len(unknown):2d}/{len(self.space.objects)} "
                f"→ {action.name}"
                + (f" {data}" if data else "")
            )

        return action, data

    def on_level_complete(self) -> None:
        """
        Called when a level ends (WIN or GAME_OVER).
        Serialises learned knowledge into cross-level priors.
        """
        self._current_stats.objects_known = len(
            [e for e in self.space.objects.values() if e.confidence >= self.confidence_threshold]
        )
        self._level_stats.append(self._current_stats)
        # Reset so _current_stats is never double-counted
        self._current_stats = LevelStats(level=self._level + 1)

        # Build priors from current embedding space
        for entry in self.space.objects.values():
            key = (entry.properties["color"], entry.properties["shape_hash"])
            existing = self._priors.get(key)
            # Keep the higher-confidence version
            if existing is None or entry.confidence > existing.confidence:
                self._priors[key] = ObjectPrior(
                    color=entry.properties["color"],
                    shape_hash=entry.properties["shape_hash"],
                    meaning_vector=entry.meaning_vector.clone(),
                    confidence=entry.confidence,
                )

        if self.verbose:
            s = self._current_stats
            log.info(
                f"Level {s.level} complete: "
                f"{s.actions_taken} actions, "
                f"{s.exploration_steps} explore / {s.planning_steps} plan, "
                f"{len(self._priors)} priors saved"
            )

    def print_stats(self) -> None:
        print(f"\n=== Agent Stats (levels played: {len(self._level_stats)}) ===")
        for s in self._level_stats:
            print(
                f"  Level {s.level}: {s.actions_taken:4d} actions  "
                f"explore={s.exploration_steps:3d}  plan={s.planning_steps:3d}  "
                f"known={s.objects_known}"
            )
        print(f"  Priors carried: {len(self._priors)}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_interaction(self, after_blobs: List[ObjectBlob]) -> None:
        target_id = self._prev_target_id
        delta = compute_delta(target_id, self._prev_blobs, after_blobs)
        self.space.update_meaning(target_id, delta[:MEANING_DIM])

        target_entry = self.space.get_entry(target_id)
        if target_entry is None:
            return

        other_ids = [oid for oid in self.space.object_ids() if oid != target_id]
        if not other_ids:
            return

        other_entry = self.space.get_entry(other_ids[0])
        if other_entry is None:
            return

        self.memory.record(InteractionRecord(
            obj_a_embed=torch.cat([target_entry.slot, target_entry.meaning_vector]),
            action_id=self._prev_action.value - 1,
            obj_b_embed=torch.cat([other_entry.slot, other_entry.meaning_vector]),
            effect_embed=delta[:EFFECT_DIM],
        ))

    def _apply_priors(self) -> None:
        """Seed known objects from previous levels with their earned meanings."""
        if not self._priors:
            return
        applied = 0
        for oid, entry in self.space.objects.items():
            key = (entry.properties["color"], entry.properties["shape_hash"])
            prior = self._priors.get(key)
            if prior and prior.confidence > 0:
                entry.meaning_vector = prior.meaning_vector.clone()
                entry.confidence = prior.confidence * 0.7  # slight discount
                applied += 1
        if self.verbose and applied:
            log.info(f"Applied {applied} priors from previous level(s)")


# ------------------------------------------------------------------
# Game runner
# ------------------------------------------------------------------

def run_game(
    game_id: str,
    max_steps_per_level: int = 300,
    verbose: bool = True,
) -> dict:
    import arc_agi

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)
    if env is None:
        raise RuntimeError(f"Could not create environment {game_id}")
    env.include_frame_data = True

    agent = ObjectCentricAgent(action_space=env.action_space, verbose=verbose)

    obs = env.reset()
    grid = obs.frame[0]
    agent.reset(grid)

    step = 0
    levels_completed = 0

    while step < max_steps_per_level * obs.win_levels:
        action, data = agent.step(grid, obs.state)

        obs = env.step(action, data=data or None)
        grid = obs.frame[0]
        step += 1

        if obs.levels_completed > levels_completed:
            levels_completed = obs.levels_completed
            agent.on_level_complete()
            if obs.state not in (GameState.WIN, GameState.GAME_OVER):
                agent.reset(grid)

        if obs.state in (GameState.WIN, GameState.GAME_OVER):
            agent.on_level_complete()
            break

    agent.print_stats()

    result = {
        "game_id": game_id,
        "total_steps": step,
        "levels_completed": levels_completed,
        "win_levels": obs.win_levels,
        "final_state": obs.state.name,
        "efficiency": levels_completed / max(step, 1),
    }

    log.info(f"Game over: {result}")
    return result


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    game_id = sys.argv[1] if len(sys.argv) > 1 else "tr87-cd924810"
    result = run_game(game_id, max_steps_per_level=200, verbose=True)
    print(f"\nFinal result: {result}")
