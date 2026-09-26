# ARC-AGI-3 object-centric agent (research prototype)

An experimental agent for [ARC-AGI-3](https://arcprize.org/), the interactive ARC benchmark. Each game is a turn-based grid environment with **no instructions, no stated rules and no stated win condition**: the agent has to explore, work out what its actions do, infer the goal, and then solve the level efficiently compared with a human baseline.

**Status: it does not solve games.** In the last full evaluation (25 games, 500 steps per level) it completed **0 levels**. This repo is the architecture, the code, and an honest account of why it failed. The improvements made after that evaluation have not been evaluated.

## Approach

Instead of feeding raw frames to a big model, the agent builds a small object-level model of the game and explores it out of curiosity:

1. **Perception** (`object_extractor.py`): connected components on the 64x64, 16-colour grid become object slots with position, colour, shape and size.
2. **Working memory** (`embedding_space.py`): a live dictionary of objects with a `meaning_vector` that starts at zero and accumulates from interaction, plus a confidence value and an "interestingness" score. An IoU tracker keeps object identities stable across frames.
3. **Curiosity-driven explorer** (`curiosity_explorer.py`): acts on the least-understood objects first. The observed change (moved, vanished, caused something else to change) is encoded as a delta vector and folded into the object's meaning. Objects that caused big reactions get explored sooner.
4. **Knowledge space** (`knowledge_space.py`): a frozen sentence-transformer table of ~78 concepts (all-MiniLM-L6-v2). A small MLP projects an object's meaning vector into that space, trained self-supervised so meanings can be labelled with a nearby concept.
5. **Relational transformer** (`relational_transformer.py`): attention over object slots to model pairwise relations, with an interaction memory. A `CausalPredictor` head predicts the effect of an action on an object pair.
6. **Goal inference and planning** (`goal_planner.py`, `react_planner.py`): heuristic goal hypotheses (bring, avoid, reach) with a greedy planner, and an optional LLM layer that reasons over the object state with a **local** model (Qwen3-4B via Ollama) and returns a goal plus a queue of actions. Without Ollama it falls back to the heuristic planner.
7. **Hybrid policy** (`agent.py`): spends at most ~30% of the step budget exploring, starts planning once ~70% of objects are understood, and forces an extra explore step when the planner repeats itself.

The full design and phased plan are in [`arc_agi3_architecture_plan.md`](arc_agi3_architecture_plan.md).

## What happened

Full evaluation on 2026-04-12, 25 games, 500 steps per level, before the later changes below: **0 levels completed.** Two failure modes:

- **Game over within ~200 steps (20 of 25 games).** The explorer clicks hazardous objects and dies. For example `tn36` spends 98% of its steps exploring and dies at step 61.
- **Never finishes (5 games: `sb26`, `lp85`, `sk48`, `sc25`, `ar25`).** The planner loops for the rest of the budget, with over 99% of steps in planning mode.

Other findings:

- In the evaluated version the relational transformer was **untrained** (random weights), so the "world model" was not actually modelling anything. That is the largest gap.
- The heuristic goal set (bring, avoid, reach) is too rigid for the variety of games.
- The hybrid policy did do what it was meant to on the exploration side: exploration dropped from ~54% to 14-31% of steps, and objects that caused reactions gained meaning and interestingness. That is a behavioural measure, not a score.
- The LLM layer is slow (about 20 s per reasoning call with Qwen3-4B) and once stalled the whole process.

## Changes made after that evaluation (not evaluated)

- `CausalPredictor` and its training step, so the relational model is trained rather than random.
- The LLM returns an open-ended goal and a queue of actions instead of picking from a fixed goal type; the agent replans when the queue empties or the score changes.
- A per-chunk streaming timeout so a stalled Ollama call cannot freeze a run.

## Next steps

- Re-run the full evaluation with the changes above and report a real number.
- Stop the explorer from clicking objects it has labelled as hazards.
- Tune the LLM call frequency and timeout against the observed latency.

## Setup and run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# optional: local LLM planner (otherwise the heuristic planner is used)
ollama pull qwen3:4b

python random_agent.py tr87-cd924810        # random-agent baseline / environment check
python agent.py tr87-cd924810               # run the agent on one game
python evaluate.py --quick                  # 3 games
python evaluate.py --record                 # all games, saving replay frames
```

Game files are fetched by the `arc-agi` package on demand into `environment_files/`, which is not committed. Tested with Python 3.12, PyTorch 2.11, arc-agi 0.9.6 and arcengine 0.9.3 on one RTX 5070 Ti.

## Note on tooling

I built this with Claude Code, and the architecture document was written as a brief for it. The design choices, the evaluation and the diagnosis of the failures are mine.
