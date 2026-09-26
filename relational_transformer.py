"""
Phase 4: Relational Transformer (World Model)

A small Transformer that operates over the set of object slots — not over
token sequences. Every step it re-reads the embedding space matrix and:

  1. Runs self-attention over [N, input_dim] object representations
  2. Produces a pairwise relation graph [N, N, relation_dim]
  3. Produces a world summary vector [summary_dim] (mean-pooled + projected)

Also maintains an InteractionMemory: a rolling buffer of
  (obj_A_embedding, action_id, obj_B_embedding, effect_embedding)
tuples from the current game.  The relational transformer attends over
this memory alongside the current object matrix.

Architecture
------------
  input_dim   = slot_dim + meaning_dim  = 96
  num_heads   = 4
  num_layers  = 2
  relation_dim = 16   (pairwise relation vector)
  summary_dim  = 64   (global world state vector)

Training signal (not wired yet, left for Phase 7+):
  Contrastive: if action A caused obj_i to affect obj_j, R[i,j] should be
  large; if no interaction was observed, R[i,j] should be small.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class WorldState:
    """Output of one RelationalTransformer forward pass."""
    relation_graph: torch.Tensor    # [N, N, relation_dim]
    world_summary: torch.Tensor     # [summary_dim]
    object_ids: List[str]           # ordered list matching dim-0 of relation_graph


@dataclass
class InteractionRecord:
    """One recorded (agent acted → world changed) tuple."""
    obj_a_embed: torch.Tensor       # [input_dim]  — the object acted on
    action_id: int
    obj_b_embed: torch.Tensor       # [input_dim]  — the object most affected
    effect_embed: torch.Tensor      # [effect_dim] — encoded delta


# ------------------------------------------------------------------
# Interaction Memory
# ------------------------------------------------------------------

EFFECT_DIM = 16   # must be ≤ meaning_dim from curiosity explorer


class InteractionMemory:
    """
    Rolling buffer of interaction records.
    Produces a fixed-size [max_records, memory_token_dim] tensor
    that the relational transformer can attend over.
    """

    def __init__(
        self,
        maxlen: int = 200,
        input_dim: int = 96,
        n_actions: int = 8,
        effect_dim: int = EFFECT_DIM,
    ):
        self.maxlen = maxlen
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.effect_dim = effect_dim
        self.memory_token_dim = input_dim + n_actions + effect_dim
        self._buffer: deque = deque(maxlen=maxlen)

    def record(self, rec: InteractionRecord) -> None:
        self._buffer.append(rec)

    def to_tensor(self) -> Optional[torch.Tensor]:
        """Returns [K, memory_token_dim] or None if buffer is empty."""
        if not self._buffer:
            return None
        rows = []
        for rec in self._buffer:
            action_onehot = F.one_hot(
                torch.tensor(rec.action_id, dtype=torch.long),
                num_classes=self.n_actions,
            ).float()
            # Pad effect to effect_dim
            eff = rec.effect_embed[:self.effect_dim]
            if eff.shape[0] < self.effect_dim:
                eff = F.pad(eff, (0, self.effect_dim - eff.shape[0]))
            token = torch.cat([rec.obj_a_embed, action_onehot, eff], dim=0)
            rows.append(token)
        return torch.stack(rows, dim=0)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# ------------------------------------------------------------------
# Causal Predictor
# ------------------------------------------------------------------

class CausalPredictor(nn.Module):
    """
    Predicts the effect of an action on a pair of objects.

    Input:  concat(obj_a[input_dim], action_onehot[n_actions], obj_b[input_dim])
    Output: predicted_effect[effect_dim]

    Trained online via MSE against actual effect_embed from InteractionMemory.
    Once trained, the weights in R[i,j] reflect real causal structure
    rather than random noise.
    """

    def __init__(
        self,
        input_dim: int = 96,
        n_actions: int = 8,
        effect_dim: int = EFFECT_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        in_features = input_dim * 2 + n_actions
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, effect_dim),
        )

    def forward(
        self,
        obj_a: torch.Tensor,      # [B, input_dim]
        action: torch.Tensor,     # [B, n_actions] one-hot
        obj_b: torch.Tensor,      # [B, input_dim]
    ) -> torch.Tensor:            # [B, effect_dim]
        x = torch.cat([obj_a, action, obj_b], dim=-1)
        return self.net(x)


# ------------------------------------------------------------------
# Relational Transformer
# ------------------------------------------------------------------

class RelationalTransformer(nn.Module):
    """
    Small transformer over object slots.

    Forward pass:
      object_matrix : [N, input_dim]
      memory_tensor : [K, memory_token_dim] or None

    Returns WorldState.
    """

    def __init__(
        self,
        input_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        relation_dim: int = 16,
        summary_dim: int = 64,
        memory_token_dim: int = 120,    # input_dim(96) + n_actions(8) + effect_dim(16)
        n_actions: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.relation_dim = relation_dim
        self.summary_dim = summary_dim
        self.n_actions = n_actions

        # Causal predictor — trained online on InteractionMemory
        self.causal = CausalPredictor(input_dim=input_dim, n_actions=n_actions, effect_dim=EFFECT_DIM)
        self._causal_optimizer = Adam(self.causal.parameters(), lr=1e-3)
        self._train_buffer: deque = deque(maxlen=2000)

        # Project object embeddings to transformer working dim
        # (keep same dim, just normalise)
        self.input_norm = nn.LayerNorm(input_dim)

        # Project memory tokens to input_dim so they can join the sequence
        self.memory_proj = nn.Linear(memory_token_dim, input_dim)

        # Transformer encoder (objects + memory tokens as sequence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pairwise relation head: concat(obj_i, obj_j) → relation vector
        self.relation_head = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, relation_dim),
        )

        # World summary head: mean pool → projection
        self.summary_head = nn.Sequential(
            nn.Linear(input_dim, summary_dim),
            nn.ReLU(),
            nn.Linear(summary_dim, summary_dim),
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        object_matrix: torch.Tensor,
        object_ids: List[str],
        memory_tensor: Optional[torch.Tensor] = None,
    ) -> WorldState:
        """
        Parameters
        ----------
        object_matrix : [N, input_dim]
        object_ids    : list of N object IDs (for labelling the output)
        memory_tensor : [K, memory_token_dim] or None

        Returns
        -------
        WorldState
        """
        N = object_matrix.shape[0]

        # Normalise object inputs
        x = self.input_norm(object_matrix)   # [N, input_dim]

        # Prepend memory tokens to the sequence (if any)
        if memory_tensor is not None and len(memory_tensor) > 0:
            mem = self.memory_proj(memory_tensor)  # [K, input_dim]
            seq = torch.cat([mem, x], dim=0)       # [K+N, input_dim]
        else:
            seq = x                                 # [N, input_dim]

        # Add batch dim
        seq = seq.unsqueeze(0)                      # [1, K+N, input_dim]

        # Transformer
        out = self.transformer(seq).squeeze(0)      # [K+N, input_dim]

        # Extract just the object tokens (last N)
        obj_out = out[-N:]                           # [N, input_dim]

        # Pairwise relation graph
        # For each (i, j) pair, concat their output embeddings
        obj_i = obj_out.unsqueeze(1).expand(-1, N, -1)   # [N, N, input_dim]
        obj_j = obj_out.unsqueeze(0).expand(N, -1, -1)   # [N, N, input_dim]
        pairs = torch.cat([obj_i, obj_j], dim=-1)         # [N, N, 2*input_dim]

        relation_graph = self.relation_head(pairs)         # [N, N, relation_dim]

        # World summary: mean pool object outputs
        world_summary = self.summary_head(obj_out.mean(dim=0))  # [summary_dim]

        return WorldState(
            relation_graph=relation_graph,
            world_summary=world_summary,
            object_ids=object_ids,
        )

    # ------------------------------------------------------------------

    def train_step(self, memory: InteractionMemory) -> float:
        """
        Sample a batch from InteractionMemory and train the CausalPredictor.

        The predictor learns: given (obj_a, action, obj_b) → predict effect.
        Once trained, the relation weights R[i,j] from the transformer encoder
        start reflecting real causal structure rather than random noise.

        Returns mean MSE loss (0.0 if buffer too small).
        """
        if len(memory) < 8:
            return 0.0

        records = list(memory._buffer)
        batch = random.sample(records, min(32, len(records)))

        obj_a = torch.stack([r.obj_a_embed for r in batch])        # [B, input_dim]
        obj_b = torch.stack([r.obj_b_embed for r in batch])        # [B, input_dim]
        effects = torch.stack([r.effect_embed[:EFFECT_DIM] for r in batch])  # [B, effect_dim]

        action_ids = torch.tensor([r.action_id for r in batch], dtype=torch.long)
        action_oh = F.one_hot(action_ids, num_classes=self.n_actions).float()  # [B, n_actions]

        pred = self.causal(obj_a, action_oh, obj_b)
        loss = F.mse_loss(pred, effects)

        self._causal_optimizer.zero_grad()
        loss.backward()
        self._causal_optimizer.step()

        return loss.item()


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import arc_agi
    import random
    from object_extractor import ObjectExtractor
    from embedding_space import EmbeddingSpace
    from curiosity_explorer import CuriosityExplorer, MEANING_DIM

    SLOT_DIM = 64
    MEANING_DIM_ = MEANING_DIM   # 32
    INPUT_DIM = SLOT_DIM + MEANING_DIM_  # 96

    arc = arc_agi.Arcade()
    env = arc.make("tr87-cd924810", render_mode=None)
    env.include_frame_data = True

    extractor = ObjectExtractor(slot_dim=SLOT_DIM, min_size=2)
    space = EmbeddingSpace(slot_dim=SLOT_DIM, meaning_dim=MEANING_DIM_)
    explorer = CuriosityExplorer(action_space=env.action_space)
    memory = InteractionMemory(maxlen=200, input_dim=INPUT_DIM)
    rt = RelationalTransformer(input_dim=INPUT_DIM)

    obs = env.reset()
    before_blobs = extractor(obs.frame[0])
    space.update(before_blobs)

    print("Running 20 steps with curiosity explorer + relational transformer...")
    for step in range(20):
        action, data, target_id = explorer.select_action(space)
        if action is None:
            action = random.choice(env.action_space)

        obs = env.step(action, data=data or None)
        after_blobs = extractor(obs.frame[0])
        space.update(after_blobs)

        if target_id is not None:
            delta = explorer.record_interaction(target_id, before_blobs, after_blobs, space)

            # Record interaction in memory if we can find both objects
            target_entry = space.get_entry(target_id)
            if target_entry is not None:
                obj_a_embed = torch.cat([target_entry.slot, target_entry.meaning_vector])
                # Find most-changed other object as obj_b (simplified: pick first other)
                other_ids = [oid for oid in space.object_ids() if oid != target_id]
                if other_ids:
                    other_entry = space.get_entry(other_ids[0])
                    if other_entry is not None:
                        obj_b_embed = torch.cat([other_entry.slot, other_entry.meaning_vector])
                        memory.record(InteractionRecord(
                            obj_a_embed=obj_a_embed,
                            action_id=action.value - 1,
                            obj_b_embed=obj_b_embed,
                            effect_embed=delta[:EFFECT_DIM],
                        ))

        before_blobs = after_blobs

    # Run the relational transformer
    mat = space.to_matrix()
    oids = space.object_ids()
    mem_tensor = memory.to_tensor()

    print(f"\nObject matrix: {mat.shape}")
    print(f"Memory buffer: {len(memory)} records → tensor {mem_tensor.shape if mem_tensor is not None else None}")

    with torch.no_grad():
        world_state = rt(mat, oids, mem_tensor)

    print(f"Relation graph: {world_state.relation_graph.shape}")
    print(f"World summary:  {world_state.world_summary.shape}")
    print(f"Summary norm:   {world_state.world_summary.norm().item():.3f}")
    print(f"Relation graph sample R[0,1]: {world_state.relation_graph[0, 1].numpy().round(3)}")
    print(f"Relation graph sample R[1,0]: {world_state.relation_graph[1, 0].numpy().round(3)}")
    print("\nPhase 4 complete.")
