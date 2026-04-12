"""
Phase 0: Random Agent
Verifies the ARC-AGI-3 environment loop works end-to-end.

Observation format:
  - obs.frame: list of one (64, 64) int8 array, values 0-15 (color indices)
  - obs.state: GameState enum (NOT_FINISHED, WIN, GAME_OVER)
  - obs.levels_completed: int
  - obs.win_levels: int (levels needed to win the game)
  - obs.available_actions: list of int action ids

Action space:
  - GameAction.ACTION1..ACTION8 (varies per game)
  - Complex actions (click-based) require {"x": int, "y": int} data

Usage:
  python random_agent.py [game_id]
  python random_agent.py tr87-cd924810        # keyboard-only game
  python random_agent.py su15-4c352900        # click-based game
"""

import sys
import random
import numpy as np

import arc_agi
from arcengine import GameAction, GameState


def run_random_agent(game_id: str, max_steps: int = 500) -> dict:
    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)
    if env is None:
        raise RuntimeError(f"Failed to create environment for {game_id}")

    env.include_frame_data = True

    obs = env.reset()
    grid = obs.frame[0]

    print(f"Game:        {game_id}")
    print(f"Grid shape:  {grid.shape}, dtype={grid.dtype}")
    print(f"Unique colors on first frame: {np.unique(grid).tolist()}")
    print(f"Win condition: complete {obs.win_levels} levels")
    print(f"Action space: {env.action_space}")
    print()

    step = 0
    levels_completed = 0

    while step < max_steps:
        # Pick a random action from the environment's action space
        action = random.choice(env.action_space)

        # Complex actions (click-based) need x, y coordinates
        action_data = {}
        if action.is_complex():
            action_data = {
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            }

        obs = env.step(action, data=action_data if action_data else None)
        step += 1

        if obs.levels_completed > levels_completed:
            levels_completed = obs.levels_completed
            grid = obs.frame[0]
            print(f"  Step {step:4d}: Level {levels_completed}/{obs.win_levels} complete "
                  f"| colors on grid: {np.unique(grid).tolist()}")

        if obs.state == GameState.WIN:
            print(f"\nWIN at step {step} — all {obs.win_levels} levels completed!")
            break
        if obs.state == GameState.GAME_OVER:
            print(f"\nGAME OVER at step {step} after {levels_completed}/{obs.win_levels} levels.")
            break
    else:
        print(f"\nReached max_steps={max_steps} with {levels_completed}/{obs.win_levels} levels.")

    result = {
        "game_id": game_id,
        "steps": step,
        "levels_completed": levels_completed,
        "win_levels": obs.win_levels,
        "final_state": str(obs.state),
    }
    print(f"\nResult: {result}")
    return result


if __name__ == "__main__":
    game_id = sys.argv[1] if len(sys.argv) > 1 else "tr87-cd924810"
    run_random_agent(game_id)
