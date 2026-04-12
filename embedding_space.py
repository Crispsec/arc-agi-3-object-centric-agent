"""
Phase 2: Real-Time Embedding Space (Working Memory)

A structured Python store — not a neural network — that tracks every
currently-visible object across frames.

Each entry:
  ObjectEntry {
      object_id     : str
      slot          : torch.Tensor [slot_dim]       — visual embedding
      properties    : dict                           — color, centroid, bbox, size
      meaning_vector: torch.Tensor [meaning_dim]    — starts at zero, accretes meaning
      confidence    : float                          — 0.0 = fully unknown, 1.0 = known
      age           : int                            — steps since first seen
      last_seen     : int                            — step index of last observation
  }

Key invariants:
  - New objects: meaning_vector = zeros, confidence = 0.0
  - Unknown = zero is the learning signal (never initialise with noise)
  - Objects not seen for `fade_after` steps lose confidence
  - Objects not seen for `remove_after` steps are dropped

IoU-based object tracker: objects in frame T are matched to frame T-1 by
overlap > 50%. Unmatched objects in T are new; unmatched from T-1 are gone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from object_extractor import ObjectBlob


# ------------------------------------------------------------------
# Data structure
# ------------------------------------------------------------------

@dataclass
class ObjectEntry:
    object_id: str
    slot: torch.Tensor                  # [slot_dim]
    properties: dict                    # color, centroid, bbox, size
    meaning_vector: torch.Tensor        # [meaning_dim], starts zeros
    confidence: float = 0.0
    age: int = 0
    last_seen: int = 0


# ------------------------------------------------------------------
# IoU tracker helper
# ------------------------------------------------------------------

def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / (float(union) + 1e-8)


def match_blobs(
    prev_blobs: List[ObjectBlob],
    curr_blobs: List[ObjectBlob],
    iou_threshold: float = 0.5,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Match curr_blobs to prev_blobs by IoU.

    Returns
    -------
    matches : {curr_blob.object_id → prev_blob.object_id}
    new_ids : [curr_blob.object_id]   — blobs in curr with no match
    gone_ids: [prev_blob.object_id]   — blobs in prev with no match
    """
    matches: Dict[str, str] = {}
    matched_prev: set = set()
    matched_curr: set = set()

    # Build IoU matrix
    if prev_blobs and curr_blobs:
        for c in curr_blobs:
            best_iou = iou_threshold
            best_p = None
            for p in prev_blobs:
                if p.object_id in matched_prev:
                    continue
                # Quick reject: color must match
                if c.color != p.color:
                    continue
                iou = _mask_iou(c.mask, p.mask)
                if iou > best_iou:
                    best_iou = iou
                    best_p = p
            if best_p is not None:
                matches[c.object_id] = best_p.object_id
                matched_prev.add(best_p.object_id)
                matched_curr.add(c.object_id)

    new_ids = [c.object_id for c in curr_blobs if c.object_id not in matched_curr]
    gone_ids = [p.object_id for p in prev_blobs if p.object_id not in matched_prev]
    return matches, new_ids, gone_ids


# ------------------------------------------------------------------
# Embedding Space
# ------------------------------------------------------------------

class EmbeddingSpace:
    """
    Working memory: tracks all currently-active objects.

    Parameters
    ----------
    slot_dim : int
        Dimension of the slot embedding from ObjectExtractor.
    meaning_dim : int
        Dimension of the meaning vector (accreted from interaction).
    fade_after : int
        Steps without observation before confidence starts decaying.
    remove_after : int
        Steps without observation before object is removed.
    confidence_per_interaction : float
        How much confidence increases per interaction delta update.
    """

    def __init__(
        self,
        slot_dim: int = 64,
        meaning_dim: int = 32,
        fade_after: int = 5,
        remove_after: int = 20,
        confidence_per_interaction: float = 0.5,
    ):
        self.slot_dim = slot_dim
        self.meaning_dim = meaning_dim
        self.fade_after = fade_after
        self.remove_after = remove_after
        self.confidence_per_interaction = confidence_per_interaction

        self.objects: Dict[str, ObjectEntry] = {}
        self._step: int = 0
        self._prev_blobs: List[ObjectBlob] = []

    # ------------------------------------------------------------------
    # Update from a new frame
    # ------------------------------------------------------------------

    def update(self, blobs: List[ObjectBlob]) -> dict:
        """
        Ingest a new frame's blob list, update tracking state.

        Returns a dict with:
          'new'       : list of new object_ids
          'updated'   : list of re-observed object_ids (current frame IDs)
          'faded'     : list of object_ids whose confidence decreased
          'removed'   : list of object_ids that were dropped
          'old_to_new': {prev_id → curr_id} for objects whose ID changed due
                        to movement (so callers can update stored ID references)
        """
        self._step += 1
        report: dict = {"new": [], "updated": [], "faded": [], "removed": [], "old_to_new": {}}

        # matches: curr_id → prev_id
        matches, new_ids, gone_ids = match_blobs(self._prev_blobs, blobs)

        # Re-observed objects: update slot + properties, keep meaning
        for blob in blobs:
            prev_id = matches.get(blob.object_id)
            if prev_id is not None and prev_id in self.objects:
                entry = self.objects[prev_id]
                entry.slot = blob.slot
                entry.properties = _properties(blob)
                entry.age += 1
                entry.last_seen = self._step

                # If the object_id hash changed due to movement, rename the key
                if prev_id != blob.object_id:
                    self.objects[blob.object_id] = entry
                    del self.objects[prev_id]
                    entry.object_id = blob.object_id
                    report["old_to_new"][prev_id] = blob.object_id

                report["updated"].append(blob.object_id)
            else:
                # New object
                entry = ObjectEntry(
                    object_id=blob.object_id,
                    slot=blob.slot,
                    properties=_properties(blob),
                    meaning_vector=torch.zeros(self.meaning_dim),
                    confidence=0.0,
                    age=0,
                    last_seen=self._step,
                )
                self.objects[blob.object_id] = entry
                report["new"].append(blob.object_id)

        # Fade / remove objects not seen this frame
        to_remove = []
        for oid, entry in self.objects.items():
            if entry.last_seen < self._step:
                unseen_for = self._step - entry.last_seen
                if unseen_for >= self.remove_after:
                    to_remove.append(oid)
                elif unseen_for >= self.fade_after:
                    entry.confidence = max(0.0, entry.confidence - 0.05)
                    report["faded"].append(oid)

        for oid in to_remove:
            del self.objects[oid]
            report["removed"].append(oid)

        self._prev_blobs = blobs
        return report

    # ------------------------------------------------------------------
    # Meaning updates (called by CuriosityExplorer after interactions)
    # ------------------------------------------------------------------

    def update_meaning(self, object_id: str, delta: torch.Tensor) -> None:
        """
        Accumulate a delta vector into an object's meaning.
        delta: [meaning_dim]
        """
        if object_id not in self.objects:
            return
        entry = self.objects[object_id]
        entry.meaning_vector = entry.meaning_vector + delta
        entry.confidence = min(1.0, entry.confidence + self.confidence_per_interaction)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_unknown_objects(
        self, threshold: float = 0.3, min_size: int = 10
    ) -> List[str]:
        """
        Object IDs sorted by confidence ascending (least known first).

        Parameters
        ----------
        threshold : float
            Objects below this confidence are considered unknown.
        min_size : int
            Only return objects with at least this many pixels.
            Tiny fragments (noise) are not worth targeting.
        """
        unknown = [
            (oid, e.confidence)
            for oid, e in self.objects.items()
            if e.confidence < threshold
            and e.properties.get("size", 0) >= min_size
        ]
        unknown.sort(key=lambda x: x[1])
        return [oid for oid, _ in unknown]

    def get_entry(self, object_id: str) -> Optional[ObjectEntry]:
        return self.objects.get(object_id)

    def to_matrix(self) -> Optional[torch.Tensor]:
        """
        Returns [N, slot_dim + meaning_dim] tensor for relational transformer.
        Returns None if no objects are tracked.
        """
        if not self.objects:
            return None
        rows = []
        for entry in self.objects.values():
            rows.append(torch.cat([entry.slot, entry.meaning_vector], dim=0))
        return torch.stack(rows, dim=0)

    def object_ids(self) -> List[str]:
        return list(self.objects.keys())

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def summarise(self) -> None:
        print(f"EmbeddingSpace at step {self._step}: {len(self.objects)} objects")
        for oid, e in sorted(self.objects.items(), key=lambda x: x[1].confidence):
            mv_norm = e.meaning_vector.norm().item()
            print(
                f"  {oid[:8]}  color={e.properties['color']:2d}  "
                f"conf={e.confidence:.2f}  age={e.age:3d}  "
                f"|meaning|={mv_norm:.3f}  "
                f"centroid=({e.properties['centroid'][0]:.1f},{e.properties['centroid'][1]:.1f})"
            )


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _properties(blob: ObjectBlob) -> dict:
    return {
        "color": blob.color,
        "centroid": blob.centroid,
        "bbox": blob.bbox,
        "size": blob.size,
        "shape_hash": blob.shape_hash,
    }


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import arc_agi
    from arcengine import GameAction
    from object_extractor import ObjectExtractor

    arc = arc_agi.Arcade()
    env = arc.make("tr87-cd924810", render_mode=None)
    env.include_frame_data = True

    extractor = ObjectExtractor(slot_dim=64, min_size=2)
    space = EmbeddingSpace(slot_dim=64, meaning_dim=32)

    obs = env.reset()
    grid = obs.frame[0]

    print("=== Frame 0 (reset) ===")
    blobs = extractor(grid)
    report = space.update(blobs)
    print(f"New: {len(report['new'])}, Updated: {len(report['updated'])}")
    space.summarise()

    print("\n=== Stepping 5 times ===")
    import random
    for i in range(5):
        action = random.choice(env.action_space)
        obs = env.step(action)
        grid = obs.frame[0]
        blobs = extractor(grid)
        report = space.update(blobs)
        print(f"Step {i+1}: new={len(report['new'])} updated={len(report['updated'])} "
              f"faded={len(report['faded'])} removed={len(report['removed'])}")

    print()
    space.summarise()

    mat = space.to_matrix()
    print(f"\nto_matrix shape: {mat.shape}")
    print(f"Unknown objects (conf<0.3): {len(space.get_unknown_objects())}")
