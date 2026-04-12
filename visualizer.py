"""
Visualizer: renders the agent's internal state at each step.

Produces a 2-panel figure per step:
  Left:  the raw 64×64 grid with object bounding boxes overlaid
  Right: agent stats — confidence histogram, top objects, active goal

Can be used in two modes:
  - live(env, agent): call inside a game loop, displays interactively
  - record(frames): saves a sequence of frames to disk as PNGs

ARC-AGI-3 color palette (indices 0-15):
  0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey,
  6=magenta, 7=orange, 8=azure, 9=maroon, 10=purple,
  11=teal, 12=navy, 13=lime, 14=pink, 15=white
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless use
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from embedding_space import EmbeddingSpace
from goal_planner import GoalHypothesis


# ARC-AGI-3 official 16-color palette (RGB 0-1)
ARC_PALETTE = [
    (0.00, 0.00, 0.00),   #  0 black
    (0.07, 0.33, 0.73),   #  1 blue
    (0.86, 0.12, 0.11),   #  2 red
    (0.07, 0.53, 0.20),   #  3 green
    (0.93, 0.83, 0.07),   #  4 yellow
    (0.55, 0.55, 0.55),   #  5 grey
    (0.78, 0.10, 0.53),   #  6 magenta
    (0.95, 0.50, 0.10),   #  7 orange
    (0.27, 0.73, 0.86),   #  8 azure
    (0.55, 0.08, 0.08),   #  9 maroon
    (0.47, 0.13, 0.63),   # 10 purple
    (0.07, 0.60, 0.53),   # 11 teal
    (0.10, 0.15, 0.45),   # 12 navy
    (0.40, 0.80, 0.15),   # 13 lime
    (0.92, 0.60, 0.73),   # 14 pink
    (1.00, 1.00, 1.00),   # 15 white
]
ARC_CMAP = ListedColormap(ARC_PALETTE)


def render_frame(
    grid: np.ndarray,
    space: EmbeddingSpace,
    step: int,
    goal: Optional[GoalHypothesis] = None,
    target_id: Optional[str] = None,
    action_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render one agent step to a matplotlib Figure.

    Parameters
    ----------
    grid       : (H, W) int8 color grid
    space      : current EmbeddingSpace state
    step       : step counter (shown in title)
    goal       : current best GoalHypothesis (or None)
    target_id  : currently targeted object ID (highlighted)
    action_name: name of the action just taken
    save_path  : if given, save the figure to this path

    Returns
    -------
    matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Step {step}  |  action={action_name}  |  "
        f"goal={goal or 'none'}  |  "
        f"known={sum(1 for e in space.objects.values() if e.confidence >= 0.3)}"
        f"/{len(space.objects)} objects",
        fontsize=9,
    )

    # ---- Left panel: grid + bounding boxes ----
    ax = axes[0]
    ax.set_title("Grid + Objects", fontsize=8)

    # Render grid using ARC palette
    rgb = np.array(ARC_PALETTE)[grid.astype(int)]  # (H, W, 3)
    ax.imshow(rgb, origin="upper", interpolation="nearest")

    for oid, entry in space.objects.items():
        r_min, c_min, r_max, c_max = entry.properties["bbox"]
        w = c_max - c_min + 1
        h = r_max - r_min + 1
        conf = entry.confidence

        color = "lime" if oid == target_id else ("yellow" if conf >= 0.3 else "white")
        lw = 2.0 if oid == target_id else 0.8
        alpha = 0.9 if oid == target_id else 0.5

        rect = patches.Rectangle(
            (c_min - 0.5, r_min - 0.5), w, h,
            linewidth=lw, edgecolor=color, facecolor="none", alpha=alpha,
        )
        ax.add_patch(rect)

        # Confidence as text in the corner of the bbox (skip tiny objects)
        if w * h >= 16:
            ax.text(
                c_min, r_min - 0.5, f"{conf:.1f}",
                fontsize=5, color=color, va="bottom", ha="left",
            )

    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)
    ax.axis("off")

    # ---- Right panel: stats ----
    ax2 = axes[1]
    ax2.axis("off")

    entries = list(space.objects.values())
    confs = [e.confidence for e in entries]
    mv_norms = [e.meaning_vector.norm().item() for e in entries]

    lines = [
        f"Step: {step}",
        f"Objects tracked: {len(entries)}",
        f"Unknown (conf<0.3): {sum(1 for c in confs if c < 0.3)}",
        f"Known (conf≥0.3): {sum(1 for c in confs if c >= 0.3)}",
        f"Mean confidence: {np.mean(confs):.3f}" if confs else "Mean confidence: n/a",
        f"Mean |meaning|: {np.mean(mv_norms):.3f}" if mv_norms else "",
        "",
        f"Goal: {goal or 'none'}",
        "",
        "Top objects by confidence:",
    ]
    top = sorted(entries, key=lambda e: -e.confidence)[:8]
    for e in top:
        marker = "→" if e.object_id == target_id else " "
        lines.append(
            f" {marker} {e.object_id[:8]} c={e.properties['color']:2d} "
            f"conf={e.confidence:.2f} |m|={e.meaning_vector.norm().item():.3f}"
        )

    text = "\n".join(lines)
    ax2.text(
        0.02, 0.98, text,
        transform=ax2.transAxes,
        fontsize=7.5, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="ghostwhite", alpha=0.8),
    )

    # Confidence histogram (inset)
    if confs:
        inset = ax2.inset_axes([0.02, 0.05, 0.5, 0.22])
        inset.hist(confs, bins=10, range=(0, 1), color="steelblue", edgecolor="white", linewidth=0.3)
        inset.set_title("conf dist", fontsize=6)
        inset.set_xlabel("confidence", fontsize=5)
        inset.tick_params(labelsize=5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig


class Recorder:
    """
    Saves a sequence of rendered frames to a directory.

    Usage:
        rec = Recorder("replays/game1")
        rec.record(grid, space, step, goal, target_id, action_name)
        rec.close()   # prints summary
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.frame_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def record(
        self,
        grid: np.ndarray,
        space: EmbeddingSpace,
        step: int,
        goal: Optional[GoalHypothesis] = None,
        target_id: Optional[str] = None,
        action_name: str = "",
    ) -> None:
        path = os.path.join(self.output_dir, f"frame_{step:04d}.png")
        fig = render_frame(grid, space, step, goal, target_id, action_name, save_path=path)
        plt.close(fig)
        self.frame_count += 1

    def close(self) -> None:
        print(f"Saved {self.frame_count} frames to {self.output_dir}/")


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import arc_agi
    import random
    from object_extractor import ObjectExtractor
    from curiosity_explorer import CuriosityExplorer, MEANING_DIM

    arc = arc_agi.Arcade()
    env = arc.make("tr87-cd924810", render_mode=None)
    env.include_frame_data = True

    extractor = ObjectExtractor(slot_dim=64, min_size=2)
    from embedding_space import EmbeddingSpace
    space = EmbeddingSpace(slot_dim=64, meaning_dim=MEANING_DIM)
    explorer = CuriosityExplorer(action_space=env.action_space)
    rec = Recorder("/tmp/arc_test_frames")

    obs = env.reset()
    blobs = extractor(obs.frame[0])
    space.update(blobs)

    for step in range(10):
        action, data, target_id = explorer.select_action(space)
        if action is None:
            action = random.choice(env.action_space)
            target_id = None

        rec.record(obs.frame[0], space, step, target_id=target_id, action_name=action.name)

        obs = env.step(action, data=data or None)
        blobs = extractor(obs.frame[0])
        space.update(blobs)

    rec.close()
    print("Visualizer smoke test passed.")
