"""
Phase 1: Object Extractor
Heuristic implementation using connected-component analysis.

Each contiguous region of the same color on the grid becomes one object.
Objects below a minimum size threshold are discarded (background noise).

Output per object:
  - object_id: str  (stable hash of color + shape)
  - color: int
  - centroid: (row, col) float
  - bbox: (row_min, col_min, row_max, col_max) int
  - size: int  (pixel count)
  - shape_hash: int  (position-invariant hash of the pixel mask)
  - mask: (H, W) bool  (pixels belonging to this object)
  - slot: torch.Tensor [slot_dim]  (learned linear projection of structured features)

The slot_dim=64 linear projection is initialised randomly but is the hook
for later training. Everything else is deterministic given the grid.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import hashlib
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import label


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class ObjectBlob:
    """One extracted object from the grid."""
    object_id: str
    color: int
    centroid: Tuple[float, float]          # (row, col)
    bbox: Tuple[int, int, int, int]        # (r_min, c_min, r_max, c_max)
    size: int
    shape_hash: int
    mask: np.ndarray                        # (H, W) bool
    # Filled in by ObjectExtractor.forward after linear projection:
    slot: torch.Tensor = field(default=None)

    @property
    def feature_vector(self) -> np.ndarray:
        """12-dim structured feature vector (no learned params)."""
        H, W = self.mask.shape
        r_min, c_min, r_max, c_max = self.bbox
        bbox_h = r_max - r_min + 1
        bbox_w = c_max - c_min + 1
        return np.array([
            self.centroid[0] / H,           # normalised row centroid
            self.centroid[1] / W,           # normalised col centroid
            self.size / (H * W),            # relative area
            bbox_h / H,                     # relative bbox height
            bbox_w / W,                     # relative bbox width
            bbox_h / (bbox_w + 1e-6),       # aspect ratio
            float(self.color) / 15.0,       # normalised color id
            # one-hot color (4 bits, colors 0-15 encoded as 4-bit binary)
            float((self.color >> 3) & 1),
            float((self.color >> 2) & 1),
            float((self.color >> 1) & 1),
            float(self.color & 1),
            float(self.shape_hash % 256) / 255.0,  # shape fingerprint
        ], dtype=np.float32)


# ------------------------------------------------------------------
# Shape hash helper
# ------------------------------------------------------------------

def _shape_hash(mask: np.ndarray) -> int:
    """Position-invariant hash: crop the bounding box, hash the binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    cropped = mask[rows][:, cols]
    return int(hashlib.md5(cropped.tobytes()).hexdigest()[:8], 16)


# ------------------------------------------------------------------
# Object ID helper
# ------------------------------------------------------------------

def _make_object_id(color: int, shape_hash: int, centroid: Tuple[float, float]) -> str:
    """
    Stable ID based on color + shape + coarse centroid bucket.
    Objects that look identical but live in different grid regions get different IDs.
    Coarse bucketing (8×8) gives some tolerance for 1-2px movement.
    """
    row_bucket = int(centroid[0] / 8)
    col_bucket = int(centroid[1] / 8)
    raw = f"{color}_{shape_hash}_{row_bucket}_{col_bucket}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ------------------------------------------------------------------
# Main extractor
# ------------------------------------------------------------------

class ObjectExtractor(nn.Module):
    """
    Heuristic object extractor.

    Takes a (H, W) int8 grid and returns a list of ObjectBlob instances,
    each with a `slot` tensor of shape [slot_dim].

    Parameters
    ----------
    slot_dim : int
        Output embedding dimension for each object.
    min_size : int
        Minimum pixel count to count as an object (filters background noise).
    background_color : int or None
        If set, regions of this color are never extracted as objects.
        None means extract all colors.
    """

    FEATURE_DIM = 12  # must match ObjectBlob.feature_vector length

    def __init__(
        self,
        slot_dim: int = 64,
        min_size: int = 2,
        background_color: int | None = None,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.min_size = min_size
        self.background_color = background_color

        # Learned linear projection: structured features → slot embedding
        self.projection = nn.Linear(self.FEATURE_DIM, slot_dim)

    # ------------------------------------------------------------------

    def forward(self, grid: np.ndarray) -> List[ObjectBlob]:
        """
        Parameters
        ----------
        grid : np.ndarray, shape (H, W), dtype int8 or int
            Raw color grid from the environment.

        Returns
        -------
        List[ObjectBlob]
            One entry per detected object, sorted by size descending.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D"
        H, W = grid.shape

        blobs: List[ObjectBlob] = []

        for color in np.unique(grid):
            if self.background_color is not None and color == self.background_color:
                continue

            color_mask = (grid == color)
            labeled, num_features = label(color_mask)

            for component_id in range(1, num_features + 1):
                comp_mask = labeled == component_id
                size = int(comp_mask.sum())
                if size < self.min_size:
                    continue

                # Bounding box
                rows, cols = np.where(comp_mask)
                r_min, r_max = int(rows.min()), int(rows.max())
                c_min, c_max = int(cols.min()), int(cols.max())

                # Centroid
                centroid = (float(rows.mean()), float(cols.mean()))

                sh = _shape_hash(comp_mask)
                oid = _make_object_id(int(color), sh, centroid)

                blob = ObjectBlob(
                    object_id=oid,
                    color=int(color),
                    centroid=centroid,
                    bbox=(r_min, c_min, r_max, c_max),
                    size=size,
                    shape_hash=sh,
                    mask=comp_mask,
                )
                blobs.append(blob)

        # Project structured features → slot embeddings
        if blobs:
            feats = torch.tensor(
                np.stack([b.feature_vector for b in blobs]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                slots = self.projection(feats)   # [N, slot_dim]

            for blob, slot in zip(blobs, slots):
                blob.slot = slot

        # Sort by size descending (largest objects first)
        blobs.sort(key=lambda b: b.size, reverse=True)
        return blobs


# ------------------------------------------------------------------
# Quick visual summary
# ------------------------------------------------------------------

def summarise(blobs: List[ObjectBlob]) -> None:
    print(f"Extracted {len(blobs)} objects:")
    for b in blobs:
        print(
            f"  id={b.object_id[:8]}  color={b.color:2d}  "
            f"size={b.size:5d}  centroid=({b.centroid[0]:5.1f},{b.centroid[1]:5.1f})  "
            f"bbox={b.bbox}  slot_norm={b.slot.norm().item():.3f}"
        )


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import arc_agi

    arc = arc_agi.Arcade()
    env = arc.make("tr87-cd924810", render_mode=None)
    env.include_frame_data = True
    obs = env.reset()

    grid = obs.frame[0]
    print(f"Grid shape: {grid.shape}, unique colors: {np.unique(grid).tolist()}")
    print()

    extractor = ObjectExtractor(slot_dim=64, min_size=2, background_color=None)
    blobs = extractor(grid)
    summarise(blobs)

    # Verify every pixel is accounted for
    covered = np.zeros(grid.shape, dtype=bool)
    for b in blobs:
        covered |= b.mask
    total_pixels = grid.size
    covered_pixels = covered.sum()
    noise_pixels = total_pixels - covered_pixels
    print(f"\nCoverage: {covered_pixels}/{total_pixels} pixels "
          f"({noise_pixels} filtered as noise with min_size={extractor.min_size})")

    # Show how many objects per color
    from collections import Counter
    color_counts = Counter(b.color for b in blobs)
    print("\nObjects per color:", dict(sorted(color_counts.items())))
