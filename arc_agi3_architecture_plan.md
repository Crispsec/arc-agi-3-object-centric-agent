# ARC-AGI-3 Object-Centric Agent — Architecture & Implementation Plan

## Context for Claude Code

This document describes a novel AI architecture designed to tackle ARC-AGI-3, the interactive benchmark where humans score 100% and frontier AI scores ~0.3%. You are helping implement a research prototype of this architecture from scratch.

ARC-AGI-3 presents turn-based game environments with no instructions, no stated rules, and no stated win condition. The agent must explore, infer the rules, discover the goal, and solve the game efficiently — compared against human first-try efficiency. Standard LLMs score near zero because they have no mechanism for active belief revision or goal inference from interaction.

---

## The Core Architecture Vision

The architecture has five components working in a continuous loop. Read all of them before writing any code.

### Component 1: Visual Perception → Object Extractor

Takes the raw 64×64 grid (16 colors) from the ARC-AGI-3 environment and extracts discrete objects into separate representations. This is **not** one big image embedding — it produces N separate object slots, one per distinct object or entity on the grid.

- Based on **Slot Attention** (Locatello et al., 2020) — each slot competes via attention to represent one object
- Output: a set of object slots `{s_1, s_2, ..., s_N}`, each a fixed-size vector (e.g. 64-dim)
- Also outputs object properties: position, color, shape, size as structured features alongside the learned slot embedding
- Objects that don't change between frames can be cached; only update slots for changed regions

### Component 2: Real-Time Embedding Space (Working Memory)

A lightweight dictionary of currently active objects and their embeddings. This is the "blackboard" — all other components read from and write to this space.

- Each object has: `{slot_embedding, properties, meaning_vector, confidence, age}`
- **Key design decision**: objects not yet interacted with have `meaning_vector = zeros` and `confidence = 0.0`. They exist but have no semantic meaning yet.
- This is NOT a neural network — it is a structured Python dict/store that updates each step
- Has a recency mechanism: objects not seen for N frames fade (reduce confidence), objects not seen for M frames are removed
- Parallel to this is a **Knowledge Space**: a frozen pretrained embedding store (can be a small sentence-transformer or lookup table) that the real-time space can query. When a meaning vector is being formed, it can be anchored to the nearest neighbor in the knowledge space. This gives the agent prior knowledge without contaminating the fresh exploration signal.

### Component 3: Curiosity-Driven Action Model (Explorer)

When objects in the real-time space have `confidence < threshold` (meaning unknown), this module takes over and generates exploratory actions targeting those objects.

- Exploration strategy: act on the lowest-confidence object first
- Action is semi-random but targeted: move toward unknown object, interact with it, observe result
- After action: the environment returns the next grid state
- The delta (what changed) is computed and fed back as a learning signal to update that object's `meaning_vector`
- Meaning update rule: `meaning_vector += learning_rate * delta_embedding` where delta_embedding encodes what happened (object moved, changed color, disappeared, caused another object to change, etc.)
- This is the "tabula rasa → meaning accretes from interaction" loop

Concretely, each interaction produces a tuple: `(object_id, action_taken, effect_observed)`. The effect is encoded as a small vector and added to the object's meaning. After a few interactions, the agent has a meaningful representation of what each object does.

### Component 4: Relational Transformer (World Model)

Takes all objects currently in the real-time embedding space — including their meaning vectors — and computes pairwise relationships between them. This is the reasoning core.

- Architecture: a small Transformer operating over the set of object slots (not over tokens)
- Input: matrix of shape `[N_objects × embedding_dim]`
- Each attention head computes: "given object A's meaning, what is its relationship to object B?"
- Outputs a **relation graph**: `R[i,j]` = relationship vector between object i and object j
- Also maintains a **history** of observed interactions: when A touched B, X happened. This is appended to the relation graph as a memory.
- The relational transformer produces a **world state summary**: a single vector encoding the agent's current understanding of the environment's rules

This component is always running — every step it re-reads the embedding space and updates the relation graph. It is the equivalent of the "LLM transformer but for objects" from the architecture vision.

### Component 5: Goal-Conditioned Planner + Action Model

Uses the world state summary from the relational transformer to infer the goal and steer actions.

**Goal inference**: The planner maintains a set of goal hypotheses, each with a probability. As the agent explores and the relational transformer reveals structure (e.g. "object A always moves toward object D when action X is taken"), hypotheses are updated. The highest-probability hypothesis becomes the active goal.

Goal hypotheses are simple structures like:
- "Get object A to position P"
- "Avoid object B touching object A"
- "Use object C to unlock object D"

**Planning**: Once a goal is active, the planner uses the relation graph to find the action sequence. This can be a simple graph search (BFS/A*) over the object relation graph, or a learned policy conditioned on the goal vector.

**Action output**: A discrete action from the ARC-AGI-3 action space (move, interact, etc.)

---

## The Full Loop (one timestep)

```
1. Receive grid from environment
2. Object Extractor → extract N object slots
3. Update Real-Time Embedding Space (new objects get zero meaning)
4. IF any objects have low confidence:
     Curiosity Action Model generates exploratory action → go to step 8
5. ELSE:
     Relational Transformer reads all objects → updates relation graph
6. Goal inference from relation graph → update goal hypotheses
7. Planner selects action toward highest-confidence goal
8. Execute action → receive new grid
9. Compute delta → update meaning vectors of affected objects
10. Go to step 1
```

---

## Implementation Plan

### Phase 0: Environment Setup

Set up the ARC-AGI-3 SDK and verify the environment interface works.

```bash
pip install arc-agi-toolkit  # or clone from github.com/arcprize/arc-agi-toolkit
```

- Get an ARC API key from arcprize.org
- Run `arc-agi --check` to verify connectivity
- Run `arc-agi --list-games` to see available environments
- Write a minimal random agent that just submits random actions to confirm the loop works
- Understand the exact observation format: 64×64 grid, 16 color integers
- Understand the exact action space: what actions are available per environment

**Deliverable**: A working random agent that completes one game (even badly).

---

### Phase 1: Object Extractor

Implement a simplified Slot Attention module.

```python
class ObjectExtractor(nn.Module):
    def __init__(self, num_slots=8, slot_dim=64, num_iterations=3):
        ...
    
    def forward(self, grid: torch.Tensor) -> SlotOutput:
        # grid: [H, W, C] one-hot encoded colors
        # returns: slots [N, slot_dim], positions [N, 2], colors [N]
```

Start simple: instead of full learned slot attention, use a heuristic extractor first.
- Connected component analysis on the grid (scipy.ndimage.label)
- Each connected region of the same color = one object
- Properties: bounding box, centroid, color, size, shape hash
- Slot embedding = learned linear projection of these properties

This gets you working objects fast. Upgrade to learned slot attention in a later phase.

**Deliverable**: Given a grid, return a list of objects with positions and property vectors.

---

### Phase 2: Real-Time Embedding Space

```python
class EmbeddingSpace:
    def __init__(self, slot_dim=64, meaning_dim=32):
        self.objects: Dict[str, ObjectEntry] = {}
    
    def upsert(self, object_id: str, slot: torch.Tensor, properties: dict):
        # If new: meaning_vector = zeros, confidence = 0.0
        # If existing: update slot, keep meaning
        ...
    
    def update_meaning(self, object_id: str, delta: torch.Tensor):
        # Accumulate meaning from interaction
        self.objects[object_id].meaning_vector += delta
        self.objects[object_id].confidence = min(1.0, conf + 0.1)
    
    def get_unknown_objects(self) -> List[str]:
        return [oid for oid, obj in self.objects.items() if obj.confidence < 0.3]
    
    def to_matrix(self) -> torch.Tensor:
        # Returns [N, slot_dim + meaning_dim] for relational transformer input
```

Object identity across frames: use a simple IoU-based tracker. If an object in frame T overlaps > 50% with an object in frame T-1, they are the same object.

**Deliverable**: Persistent object store that correctly tracks objects across frames and initializes new ones at zero.

---

### Phase 3: Curiosity-Driven Explorer

```python
class CuriosityExplorer:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def select_action(self, embedding_space: EmbeddingSpace) -> Action:
        unknown = embedding_space.get_unknown_objects()
        if not unknown:
            return None  # Hand off to planner
        target = unknown[0]  # Lowest confidence first
        return self.action_toward(target, embedding_space)
    
    def compute_delta(self, before: Grid, after: Grid, action: Action) -> Dict[str, torch.Tensor]:
        # Which objects changed? How?
        # Returns {object_id: delta_vector}
        # Delta encodes: moved (direction), changed color, appeared, disappeared,
        #                caused another object to change
```

Delta encoding: keep it structured, not learned at first. A delta vector of dim 16:
- dims 0-3: movement direction (N/S/E/W magnitude)
- dims 4-7: color change (before/after)
- dim 8: disappeared flag
- dim 9: caused_other_to_change flag
- dims 10-15: spare for future

**Deliverable**: Explorer that targets unknown objects, takes actions, and produces delta vectors that get applied to meaning embeddings.

---

### Phase 4: Relational Transformer

```python
class RelationalTransformer(nn.Module):
    def __init__(self, input_dim=96, num_heads=4, num_layers=2):
        # Small transformer: 2 layers, 4 heads
        # Input dim = slot_dim (64) + meaning_dim (32)
        ...
    
    def forward(self, object_matrix: torch.Tensor) -> WorldState:
        # object_matrix: [N, input_dim]
        # Self-attention over objects (not tokens)
        # Returns:
        #   relation_graph: [N, N, relation_dim]  pairwise relationships
        #   world_summary: [summary_dim]  global state vector
```

Training signal for this module: contrastive — if action X caused A to affect B, the relation A→B should be strong. If no interaction was observed, relation should be weak. This can be self-supervised from the interaction history.

Also maintain an **interaction memory**: a rolling buffer of `(obj_A, action, obj_B, effect)` tuples from the current game. The relational transformer attends over this memory alongside the current object matrix.

**Deliverable**: Transformer that takes N object embeddings and produces a pairwise relation graph and world summary vector.

---

### Phase 5: Goal Inference + Planner

```python
class GoalInferenceModule:
    def __init__(self):
        self.hypotheses: List[GoalHypothesis] = []
    
    def update(self, world_state: WorldState, interaction_history: List):
        # Generate new hypotheses from observed patterns
        # Update probabilities based on evidence
        # Prune low-probability hypotheses
        ...
    
    def best_goal(self) -> GoalHypothesis:
        return max(self.hypotheses, key=lambda h: h.probability)

class Planner:
    def select_action(self, goal: GoalHypothesis, 
                      relation_graph: RelationGraph,
                      embedding_space: EmbeddingSpace) -> Action:
        # BFS/A* over object graph toward goal
        # Uses relation_graph to predict action effects
        ...
```

Goal hypothesis types to start with (keep it simple):
1. "Move object X to position P" — inferred if the win signal fires after X reaches P
2. "Keep objects X and Y apart" — inferred if penalty fires when they touch
3. "Bring object X near object Y" — inferred from positive feedback patterns

The key insight: don't try to infer the goal immediately. Explore first, build the relation graph, then look for what configuration correlates with the win/lose signal.

**Deliverable**: Goal inference that generates and ranks hypotheses, and a planner that steers actions toward the best hypothesis.

---

### Phase 6: Integration + Game Loop

Wire all components into a single agent class:

```python
class ObjectCentricAgent:
    def __init__(self):
        self.extractor = ObjectExtractor()
        self.embedding_space = EmbeddingSpace()
        self.explorer = CuriosityExplorer(action_space)
        self.relational_transformer = RelationalTransformer()
        self.goal_inference = GoalInferenceModule()
        self.planner = Planner()
        self.interaction_memory = InteractionMemory(maxlen=200)
    
    def step(self, grid: np.ndarray) -> Action:
        # Full loop as described above
        ...
    
    def on_level_complete(self):
        # Compress what was learned into knowledge space
        # Keep object meanings, relation patterns
        # Carry forward to next level
```

**Cross-level transfer**: When a level ends, serialize the relation graph and object meaning vectors. At the start of the next level, initialize the embedding space with these priors. Objects recognized from level 1 start with their earned meaning rather than zero. New objects still start at zero.

**Deliverable**: Complete agent that plays a full ARC-AGI-3 game end to end.

---

### Phase 7: Evaluation + Iteration

- Run the agent against the public ARC-AGI-3 environments
- Measure: actions taken vs human baseline (efficiency score)
- Identify failure modes: is it failing at object extraction? goal inference? planning?
- Instrument each component with logging so you can see what the agent "thinks"
- Build a replay visualizer: show the grid, the extracted objects, the relation graph, and the active goal hypothesis at each step

---

## Hardware & Stack

You have an RTX 5070 Ti (16GB VRAM). This is sufficient for:
- Slot attention over 64×64 grids: trivial
- Relational transformer with N≤20 objects, dim=96: trivial
- The whole agent runs in real time, no GPU bottleneck

Stack:
- Python 3.11+
- PyTorch 2.x (CUDA)
- scipy (connected components for heuristic extractor)
- numpy
- arc-agi-toolkit (official SDK)

Keep dependencies minimal. This is a research prototype, not production.

---

## Key Design Principles

**Start heuristic, upgrade to learned**: Use connected components before slot attention. Use BFS before learned planning. Get the loop working first, then replace components with learned versions.

**Instrument everything**: Every component should log what it's doing. You need to watch the agent think to understand where it breaks.

**Per-object, not per-frame**: The fundamental unit is the object, not the grid frame. All components should operate on objects. Never pass raw grids to the relational transformer.

**Zero means unknown, not zero**: The zero embedding for unknown objects is a design choice that must be preserved throughout. Don't accidentally initialize with noise or pretrained values. Unknown = zero is the learning signal.

**Efficiency matters from day one**: Log action count at every step. The benchmark penalizes waste. Build in an "action budget awareness" from the start.

---

## What Success Looks Like

A working prototype should demonstrate:
1. Agent correctly segments the grid into objects
2. Unknown objects trigger exploration, not random flailing
3. After ~5 interactions with an object, its meaning vector is meaningfully different from zero and reflects what it does
4. The relation graph correctly captures "when A touches B, B moves"
5. The agent can form a goal hypothesis and steer toward it
6. The agent carries knowledge from level 1 into level 2

You don't need to beat the human baseline. You need to demonstrate the loop works conceptually. The architecture is the contribution — the scores come later with tuning.
