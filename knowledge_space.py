"""
Knowledge Space — concept vocabulary + self-supervised MLP projector.

Architecture:
  1. Frozen sentence-transformer embeds ~100 concept strings → [V, 384] table
  2. MLP projector maps meaning_vector[32] → concept_space[384]
  3. lookup_and_label() blends visual description + meaning projection to find
     the nearest concept for any object
  4. Self-supervised training: delta → text description → frozen sentence-transformer
     target → cosine loss on the MLP projector

The projector starts random; after a few dozen interactions it becomes
meaningful. Weights are saved to knowledge_space_projector.pt and auto-loaded
on subsequent runs.
"""

from __future__ import annotations

import collections
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding_space import ObjectEntry


# ------------------------------------------------------------------
# Concept vocabulary (~100 strings)
# ------------------------------------------------------------------

CONCEPT_VOCAB: List[str] = [
    # --- motion ---
    "moves when pushed",
    "slides continuously until wall",
    "cannot be moved",
    "bounces back",
    "moves north when touched",
    "moves south when touched",
    "moves west when touched",
    "moves east when touched",
    "teleports when touched",
    "rotates when touched",
    # --- causal ---
    "causes chain reaction when touched",
    "switches state when touched",
    "activates another object when reached",
    "pushes other objects",
    "pulls other objects",
    "destroys other objects on contact",
    "spawns new objects",
    "opens when key is nearby",
    "blocks movement",
    "triggers when stepped on",
    # --- role ---
    "player character",
    "goal or target location",
    "collectible item",
    "key object",
    "door or gate",
    "enemy or hazard",
    "wall or obstacle",
    "platform or surface",
    "switch or button",
    "projectile",
    # --- interaction outcome ---
    "disappears on contact",
    "merges with similar objects",
    "changes color when touched",
    "splits into smaller objects",
    "grows when collecting items",
    "was pushed by external force",
    "removed by interaction",
    "no observable effect",
    # --- visual: shape ---
    "small square",
    "large square",
    "thin vertical strip",
    "thin horizontal strip",
    "L-shaped object",
    "T-shaped object",
    "cross-shaped object",
    "diagonal line",
    "single pixel",
    "large irregular blob",
    # --- visual: color + role ---
    "black background element",
    "blue interactive object",
    "red dangerous object",
    "green goal or safe zone",
    "yellow collectible or key",
    "grey neutral obstacle",
    "magenta special object",
    "orange moving object",
    "cyan player or indicator",
    "white boundary or border",
    # --- composite ---
    "small red player",
    "small blue player",
    "large green target",
    "small yellow collectible",
    "large grey wall",
    "small orange enemy",
    "small cyan interactive",
    "red disappearing object",
    "blue sliding block",
    "green unlockable door",
    # --- behaviour patterns ---
    "object that enables level completion",
    "object that blocks level completion",
    "indicator or counter",
    "mirrored object",
    "paired object",
    "stationary anchor",
    "frequently interacted object",
    "rarely changes state",
    "always reacts to player",
    "independent moving object",
]

# ------------------------------------------------------------------
# ARC color names (colors 0–15)
# ------------------------------------------------------------------

_COLOR_NAMES = {
    0:  "black",
    1:  "blue",
    2:  "red",
    3:  "green",
    4:  "yellow",
    5:  "grey",
    6:  "magenta",
    7:  "orange",
    8:  "cyan",
    9:  "maroon",
    10: "purple",
    11: "teal",
    12: "brown",
    13: "lime",
    14: "navy",
    15: "white",
}


# ------------------------------------------------------------------
# Visual description helper
# ------------------------------------------------------------------

def _visual_description(entry: ObjectEntry) -> str:
    """
    Build a text description from an ObjectEntry's visual properties.
    e.g. "small red square", "large green vertical strip"
    """
    size = entry.properties.get("size", 0)
    if size < 4:
        size_word = "tiny"
    elif size < 16:
        size_word = "small"
    elif size < 64:
        size_word = "medium"
    else:
        size_word = "large"

    bbox = entry.properties.get("bbox", (0, 0, 1, 1))
    r0, c0, r1, c1 = bbox
    h = max(1, r1 - r0)
    w = max(1, c1 - c0)
    ratio = h / w
    if ratio > 2.5:
        shape_word = "vertical strip"
    elif ratio < 0.4:
        shape_word = "horizontal strip"
    elif 0.7 < ratio < 1.4:
        shape_word = "square"
    else:
        shape_word = "rectangle"

    color_name = _COLOR_NAMES.get(entry.properties.get("color", 0), "unknown")
    return f"{size_word} {color_name} {shape_word}"


# ------------------------------------------------------------------
# Delta → text description helper (for self-supervised training)
# ------------------------------------------------------------------

def _delta_to_text(delta: torch.Tensor) -> str:
    """
    Convert a 32-dim interaction delta vector to a human-readable description.
    Used to generate supervision signal for the MLP projector.
    """
    d = delta.detach().cpu().numpy()
    parts = []

    # dims 0-3: movement N/S/E/W
    if d[0] > 0.05:
        parts.append("moved north")
    if d[1] > 0.05:
        parts.append("moved south")
    if d[2] > 0.05:
        parts.append("moved west")
    if d[3] > 0.05:
        parts.append("moved east")

    # dim 8: target disappeared
    if d[8] > 0.5:
        parts.append("disappeared on contact")

    # dim 9: caused other objects to change
    if d[9] > 0.5:
        parts.append("caused other objects to change")

    # dim 10: chain reaction (many objects changed)
    if d[10] > 0.3:
        parts.append("caused chain reaction")

    # dims 11-14: reactor movement (when this object was a reactor)
    if d[11] > 0.05 or d[12] > 0.05:
        parts.append("was pushed by external force")
    if d[13] > 0.05 or d[14] > 0.05:
        parts.append("was pushed sideways")

    # dim 15: reactor disappeared
    if d[15] > 0.5:
        parts.append("was removed by interaction")

    return ", ".join(parts) if parts else "no observable effect"


# ------------------------------------------------------------------
# MLP Projector
# ------------------------------------------------------------------

class _Projector(nn.Module):
    def __init__(self, meaning_dim: int = 32, hidden: int = 128, concept_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meaning_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, concept_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ------------------------------------------------------------------
# Knowledge Space
# ------------------------------------------------------------------

class KnowledgeSpace:
    """
    Maps object meaning vectors to interpretable concepts via a concept vocabulary
    and a self-supervised MLP projector.

    Parameters
    ----------
    model_name : str
        Sentence-transformer model to use for concept + visual embeddings.
    projector_hidden : int
        Hidden dim of the MLP projector.
    meaning_dim : int
        Dimension of the input meaning vectors (must match agent's MEANING_DIM).
    concept_dim : int
        Output dim of the sentence transformer.
    blend_alpha : float
        Weight of meaning projection vs visual description embedding.
        0.0 = visual only, 1.0 = meaning only.
    projector_ckpt : str
        Path to save/load projector weights.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        projector_hidden: int = 128,
        meaning_dim: int = 32,
        concept_dim: int = 384,
        blend_alpha: float = 0.5,
        projector_ckpt: str = "knowledge_space_projector.pt",
    ):
        self.meaning_dim = meaning_dim
        self.concept_dim = concept_dim
        self.blend_alpha = blend_alpha
        self.projector_ckpt = Path(projector_ckpt)
        self._train_calls = 0

        # Load sentence transformer (cached by the library)
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(model_name)
        self._st.eval()

        # Embed concept vocabulary — frozen
        self._concept_embeddings: torch.Tensor = self._embed_texts(CONCEPT_VOCAB)  # [V, D]

        # MLP projector
        self._projector = _Projector(meaning_dim, projector_hidden, concept_dim)
        if self.projector_ckpt.exists():
            self._projector.load_state_dict(torch.load(self.projector_ckpt, weights_only=True))

        self._optimizer = torch.optim.Adam(self._projector.parameters(), lr=1e-3)

        # Training buffer
        self._train_buffer: collections.deque = collections.deque(maxlen=2000)

        # Text embedding cache (visual descriptions are repetitive)
        self._text_cache: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup_and_label(self, entry: ObjectEntry) -> Tuple[str, float]:
        """
        Find the nearest concept for an object and update its visual_concept field.

        Uses visual description alone if the meaning vector is near-zero
        (projector output is meaningless for fresh objects).
        Blends meaning projection + visual embedding otherwise.

        Returns (concept_str, similarity_score).
        """
        visual_text = _visual_description(entry)
        visual_embed = self._embed_text(visual_text)  # [D]

        mv_norm = entry.meaning_vector.norm().item()
        if mv_norm < 0.01:
            query = visual_embed
        else:
            with torch.no_grad():
                mv = entry.meaning_vector.unsqueeze(0)  # [1, 32]
                proj = self._projector(mv).squeeze(0)   # [D]
            query = F.normalize(
                (1.0 - self.blend_alpha) * visual_embed + self.blend_alpha * proj,
                dim=0,
            )

        sims = (self._concept_embeddings @ query)  # [V]
        best_idx = int(sims.argmax().item())
        best_sim = float(sims[best_idx].item())
        concept = CONCEPT_VOCAB[best_idx]

        entry.visual_concept = concept
        entry.concept_confidence = best_sim
        return concept, best_sim

    def record_training_sample(
        self,
        meaning_vector: torch.Tensor,
        delta: torch.Tensor,
    ) -> None:
        """Buffer a training sample. Call train() periodically."""
        text = _delta_to_text(delta)
        target_embed = self._embed_text(text).detach()
        self._train_buffer.append((meaning_vector.detach().clone(), target_embed))

    def train(self, steps: int = 50) -> float:
        """
        Train the projector on buffered samples (cosine loss).
        Saves checkpoint every 100 calls.
        Returns mean loss (0.0 if buffer too small).
        """
        if len(self._train_buffer) < 8:
            return 0.0

        batch = random.sample(list(self._train_buffer), min(32, len(self._train_buffer)))
        mv_batch = torch.stack([x[0] for x in batch])   # [B, 32]
        tgt_batch = torch.stack([x[1] for x in batch])  # [B, D]

        proj = self._projector(mv_batch)  # already normalised inside Projector
        loss = (1.0 - (proj * tgt_batch).sum(dim=-1)).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._train_calls += 1
        if self._train_calls % 100 == 0:
            torch.save(self._projector.state_dict(), self.projector_ckpt)

        return float(loss.item())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_text(self, text: str) -> torch.Tensor:
        """Encode a single string, using an in-memory cache."""
        if text not in self._text_cache:
            self._text_cache[text] = self._embed_texts([text]).squeeze(0)
        return self._text_cache[text]

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of strings → normalised [N, D] tensor."""
        with torch.no_grad():
            embs = self._st.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embs.float().cpu()


# ------------------------------------------------------------------
# Smoke test (no env dependency)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    print("Loading KnowledgeSpace (downloads model on first run)...")
    ks = KnowledgeSpace()
    print(f"Concept vocabulary: {len(CONCEPT_VOCAB)} entries")
    print(f"Concept embedding table: {ks._concept_embeddings.shape}")

    # Dummy ObjectEntry with near-zero meaning (visual-only path)
    from dataclasses import dataclass, field as dc_field
    entry_fresh = ObjectEntry(
        object_id="test_fresh",
        slot=torch.zeros(64),
        properties={"color": 2, "centroid": (10.0, 15.0), "bbox": (8, 12, 14, 18), "size": 36, "shape_hash": 0},
        meaning_vector=torch.zeros(32),
    )
    concept, sim = ks.lookup_and_label(entry_fresh)
    print(f"\nFresh object (visual-only path):")
    print(f"  description : {_visual_description(entry_fresh)}")
    print(f"  concept     : {concept!r}  (sim={sim:.3f})")

    # Dummy ObjectEntry with nonzero meaning (blend path)
    mv = torch.zeros(32)
    mv[1] = 0.8  # moved south
    mv[9] = 1.0  # caused others to change
    entry_known = ObjectEntry(
        object_id="test_known",
        slot=torch.zeros(64),
        properties={"color": 7, "centroid": (30.0, 30.0), "bbox": (28, 28, 33, 33), "size": 25, "shape_hash": 0},
        meaning_vector=mv,
    )
    concept2, sim2 = ks.lookup_and_label(entry_known)
    print(f"\nKnown object (blend path, projector random):")
    print(f"  description : {_visual_description(entry_known)}")
    print(f"  concept     : {concept2!r}  (sim={sim2:.3f})")

    # Training smoke test
    delta = torch.zeros(32)
    delta[1] = 0.4   # south
    delta[9] = 1.0   # caused others to change
    print(f"\nDelta text: {_delta_to_text(delta)!r}")

    ks.record_training_sample(mv, delta)
    for _ in range(10):
        ks.record_training_sample(mv + torch.randn(32) * 0.1, delta)
    loss = ks.train()
    print(f"Training loss (10 samples): {loss:.4f}")

    print("\nSmoke test passed.")
