"""
Phase 3: Curiosity-Driven Explorer

When objects in the embedding space have confidence < threshold, this module:
  1. Selects the lowest-confidence object as the exploration target
  2. Generates a targeted action toward that object
  3. After the action resolves, computes a delta vector encoding what changed
  4. Returns the delta so EmbeddingSpace.update_meaning() can be called

Delta vector (meaning_dim=32):
  dims  0- 3: movement direction (N/S/E/W) as signed pixel magnitudes, normalised
  dims  4- 7: color change for target object (before/after, each 4-bit encoded)
  dim   8   : target object disappeared (1.0) or not (0.0)
  dim   9   : target object caused another object to change (1.0) or not
  dims 10-15: other objects changed count, normalised; spare
  dims 16-31: spare (zeros for now, reserved for future interaction features)

Action targeting:
  - keyboard games: use directional actions (ACTION1=up, ACTION2=down,
    ACTION3=left, ACTION4=right if available) to move toward target
  - click games: directly click the target's centroid
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
from arcengine import GameAction

from object_extractor import ObjectBlob, ObjectExtractor
from embedding_space import EmbeddingSpace, ObjectEntry


MEANING_DIM = 32

# Directional action heuristic for keyboard games.
# We assume ACTION1/2/3/4 = up/down/left/right (common convention).
# If the game doesn't follow this convention, the explorer will still
# produce valid actions — they just won't move optimally toward the target.
_DIR_ACTIONS = {
    "up":    GameAction.ACTION1,
    "down":  GameAction.ACTION2,
    "left":  GameAction.ACTION3,
    "right": GameAction.ACTION4,
}


# ------------------------------------------------------------------
# Delta encoder
# ------------------------------------------------------------------

def compute_delta(
    target_id: str,
    before_blobs: List[ObjectBlob],
    after_blobs: List[ObjectBlob],
) -> torch.Tensor:
    """
    Encode what happened to `target_id` (and the world) as a delta vector.

    Parameters
    ----------
    target_id : str
        The object ID of the interaction target.
    before_blobs : List[ObjectBlob]
        Blobs extracted from the grid before the action.
    after_blobs : List[ObjectBlob]
        Blobs extracted from the grid after the action.

    Returns
    -------
    torch.Tensor of shape [MEANING_DIM]
    """
    delta = np.zeros(MEANING_DIM, dtype=np.float32)

    # Build id→blob maps (approximate: match by color + nearest centroid)
    before_map = {b.object_id: b for b in before_blobs}
    after_map = {b.object_id: b for b in after_blobs}

    target_before = before_map.get(target_id)
    target_after = _find_closest_same_color(target_id, before_blobs, after_blobs)

    # --- dims 0-3: movement (N/S/E/W) ---
    if target_before is not None and target_after is not None:
        dr = target_after.centroid[0] - target_before.centroid[0]
        dc = target_after.centroid[1] - target_before.centroid[1]
        # Normalise by grid size (64)
        delta[0] = max(0.0, -dr) / 64.0   # north
        delta[1] = max(0.0,  dr) / 64.0   # south
        delta[2] = max(0.0, -dc) / 64.0   # west
        delta[3] = max(0.0,  dc) / 64.0   # east

    # --- dims 4-7: color change (4-bit encoding before, 4-bit after) ---
    if target_before is not None:
        c_before = target_before.color
        delta[4] = float((c_before >> 3) & 1)
        delta[5] = float((c_before >> 2) & 1)
        delta[6] = float((c_before >> 1) & 1)
        delta[7] = float(c_before & 1)
    if target_after is not None:
        c_after = target_after.color
        # Overwrite with after color if it changed
        if target_before is None or c_after != target_before.color:
            delta[4] = float((c_after >> 3) & 1)
            delta[5] = float((c_after >> 2) & 1)
            delta[6] = float((c_after >> 1) & 1)
            delta[7] = float(c_after & 1)

    # --- dim 8: target disappeared ---
    if target_before is not None and target_after is None:
        delta[8] = 1.0

    # --- dim 9: other objects changed ---
    changed_count = _count_changed_objects(before_blobs, after_blobs, exclude_id=target_id)
    if changed_count > 0:
        delta[9] = 1.0
    # dim 10: normalised count of other changes
    delta[10] = min(1.0, changed_count / 10.0)

    return torch.tensor(delta, dtype=torch.float32)


def _find_closest_same_color(
    target_id: str,
    before_blobs: List[ObjectBlob],
    after_blobs: List[ObjectBlob],
) -> Optional[ObjectBlob]:
    """
    Find the blob in after_blobs that best corresponds to the target.
    First try exact ID match; fall back to nearest centroid with same color.
    """
    # Exact match
    for b in after_blobs:
        if b.object_id == target_id:
            return b

    # Find target in before
    target_before = next((b for b in before_blobs if b.object_id == target_id), None)
    if target_before is None:
        return None

    # Nearest same-color blob in after
    best = None
    best_dist = float("inf")
    for b in after_blobs:
        if b.color != target_before.color:
            continue
        dr = b.centroid[0] - target_before.centroid[0]
        dc = b.centroid[1] - target_before.centroid[1]
        dist = dr * dr + dc * dc
        if dist < best_dist:
            best_dist = dist
            best = b
    return best


def _count_changed_objects(
    before_blobs: List[ObjectBlob],
    after_blobs: List[ObjectBlob],
    exclude_id: str,
) -> int:
    """Count how many non-target objects changed position or disappeared."""
    before_map = {b.object_id: b for b in before_blobs if b.object_id != exclude_id}
    after_map = {b.object_id: b for b in after_blobs if b.object_id != exclude_id}

    changed = 0
    for oid, b_before in before_map.items():
        b_after = after_map.get(oid)
        if b_after is None:
            changed += 1  # disappeared
        else:
            dr = abs(b_after.centroid[0] - b_before.centroid[0])
            dc = abs(b_after.centroid[1] - b_before.centroid[1])
            if dr + dc > 1.0:
                changed += 1  # moved
    return changed


# ------------------------------------------------------------------
# Curiosity Explorer
# ------------------------------------------------------------------

class CuriosityExplorer:
    """
    Generates exploratory actions targeting unknown objects.

    When all objects are known (confidence >= threshold), returns None —
    signalling that the Planner should take over.

    Parameters
    ----------
    action_space : List[GameAction]
        Available actions for the current environment.
    confidence_threshold : float
        Objects below this are considered unknown.
    grid_size : int
        Grid dimensions (assumed square).
    """

    def __init__(
        self,
        action_space: List[GameAction],
        confidence_threshold: float = 0.3,
        grid_size: int = 64,
    ):
        self.action_space = action_space
        self.confidence_threshold = confidence_threshold
        self.grid_size = grid_size
        self._current_target_id: Optional[str] = None   # sticky target

    def notify_id_remap(self, old_to_new: dict) -> None:
        """Call after EmbeddingSpace.update() to keep the sticky target fresh."""
        if self._current_target_id in old_to_new:
            self._current_target_id = old_to_new[self._current_target_id]

    # ------------------------------------------------------------------

    def select_action(
        self,
        space: EmbeddingSpace,
    ) -> Tuple[Optional[GameAction], Optional[dict], Optional[str]]:
        """
        Choose an action targeting the least-known object.

        Sticky targeting: once a target is chosen, keep interacting with it
        until its confidence crosses the threshold, then move to the next.

        Returns
        -------
        action : GameAction or None (None = hand off to planner)
        data   : dict with "x","y" for click actions, {} otherwise
        target_id : str or None — the object being targeted
        """
        unknown = space.get_unknown_objects(self.confidence_threshold)
        if not unknown:
            self._current_target_id = None
            return None, None, None

        # Keep current target if it is still unknown; pick new one otherwise
        if (
            self._current_target_id is None
            or self._current_target_id not in space.objects
            or space.objects[self._current_target_id].confidence >= self.confidence_threshold
        ):
            self._current_target_id = unknown[0]   # lowest confidence

        target_id = self._current_target_id
        entry = space.get_entry(target_id)
        if entry is None:
            self._current_target_id = None
            return None, None, None

        action, data = self._action_toward(entry)
        return action, data, target_id

    # ------------------------------------------------------------------

    def _action_toward(
        self, entry: ObjectEntry
    ) -> Tuple[GameAction, dict]:
        """
        Pick an action that moves toward the target's centroid,
        or clicks it directly for click-based action spaces.
        """
        centroid_row, centroid_col = entry.properties["centroid"]

        # Click action: directly target centroid
        click_actions = [a for a in self.action_space if a.is_complex()]
        if click_actions:
            action = click_actions[0]
            return action, {
                "x": int(round(centroid_col)),
                "y": int(round(centroid_row)),
            }

        # Keyboard action: pick direction toward target
        # We assume the agent is implicitly "in" the grid and can navigate.
        # Since we don't know the agent's position, use the target's centroid
        # to decide the dominant direction from grid centre.
        cx = self.grid_size / 2.0
        cy = self.grid_size / 2.0
        dr = centroid_row - cy
        dc = centroid_col - cx

        direction = _dominant_direction(dr, dc)
        action = _DIR_ACTIONS.get(direction)

        # Fall back to any available action if direction action isn't in space
        if action not in self.action_space:
            import random
            action = random.choice(self.action_space)

        return action, {}

    # ------------------------------------------------------------------

    def record_interaction(
        self,
        target_id: str,
        before_blobs: List[ObjectBlob],
        after_blobs: List[ObjectBlob],
        space: EmbeddingSpace,
    ) -> torch.Tensor:
        """
        Compute the delta from an interaction and apply it to the embedding space.

        Returns the delta tensor (for logging / relational transformer memory).
        """
        delta = compute_delta(target_id, before_blobs, after_blobs)
        space.update_meaning(target_id, delta[:space.meaning_dim])
        return delta


def _dominant_direction(dr: float, dc: float) -> str:
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import arc_agi
    from object_extractor import ObjectExtractor

    arc = arc_agi.Arcade()
    env = arc.make("tr87-cd924810", render_mode=None)
    env.include_frame_data = True

    extractor = ObjectExtractor(slot_dim=64, min_size=2)
    space = EmbeddingSpace(slot_dim=64, meaning_dim=MEANING_DIM)
    explorer = CuriosityExplorer(action_space=env.action_space)

    obs = env.reset()
    grid = obs.frame[0]
    before_blobs = extractor(grid)
    space.update(before_blobs)

    print(f"Initial unknown objects: {len(space.get_unknown_objects())}")
    print()

    interaction_log = []
    for step in range(10):
        action, data, target_id = explorer.select_action(space)
        if action is None:
            print("All objects known — explorer hands off to planner.")
            break

        target_entry = space.get_entry(target_id)
        print(
            f"Step {step+1}: targeting {target_id[:8]} "
            f"(color={target_entry.properties['color']}, "
            f"conf={target_entry.confidence:.2f}, "
            f"centroid=({target_entry.properties['centroid'][0]:.1f},"
            f"{target_entry.properties['centroid'][1]:.1f})) "
            f"→ action={action.name}"
            + (f" data={data}" if data else "")
        )

        obs = env.step(action, data=data or None)
        after_blobs = extractor(obs.frame[0])
        space.update(after_blobs)

        delta = explorer.record_interaction(target_id, before_blobs, after_blobs, space)
        interaction_log.append((target_id, action, delta))

        updated_entry = space.get_entry(target_id)
        if updated_entry:
            print(
                f"         → conf now {updated_entry.confidence:.2f}, "
                f"|meaning|={updated_entry.meaning_vector.norm().item():.4f}, "
                f"delta={delta[:11].numpy().round(3)}"
            )

        before_blobs = after_blobs

    print(f"\nAfter {len(interaction_log)} interactions:")
    print(f"  Unknown objects remaining: {len(space.get_unknown_objects())}")
    confident = [e for e in space.objects.values() if e.confidence > 0]
    print(f"  Objects with conf>0: {len(confident)}")
    if confident:
        print("  Top by confidence:")
        for e in sorted(confident, key=lambda x: -x.confidence)[:5]:
            print(f"    {e.object_id[:8]} conf={e.confidence:.2f} "
                  f"|meaning|={e.meaning_vector.norm().item():.4f}")
