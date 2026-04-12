"""
Phase 7: Evaluation Harness

Runs the ObjectCentricAgent across multiple games and reports:
  - Steps taken vs. human baseline (efficiency ratio)
  - Levels completed per game
  - Explorer / planner split
  - Final state (WIN / GAME_OVER / timeout)

Usage:
  python evaluate.py                   # run all 25 available games
  python evaluate.py tr87 ls20 sb26    # run specific game prefixes
  python evaluate.py --record          # save replay frames
  python evaluate.py --quick           # 3 keyboard-only games
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import List, Optional

import arc_agi
from arcengine import GameState

from agent import ObjectCentricAgent, run_game


log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Game lists
# ------------------------------------------------------------------

# Keyboard-only games (ACTION1-4 = up/down/left/right, no click)
KEYBOARD_GAMES = [
    "tr87-cd924810",
    "ls20-9607627b",
    "g50t-5849a774",
    "wa30-ee6fef47",
]

# Click-only games
CLICK_GAMES = [
    "su15-4c352900",
    "s5i5-a48e4b1d",
    "lp85-305b61c3",
    "vc33-9851e02b",
    "r11l-aa269680",
    "tn36-ab4f63cc",
    "lf52-271a04aa",
]

QUICK_GAMES = ["tr87-cd924810", "sb26-7fbdac44", "ft09-0d8bbf25"]


# ------------------------------------------------------------------
# Single-game evaluation
# ------------------------------------------------------------------

def evaluate_game(
    game_id: str,
    max_steps_per_level: int = 300,
    record: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run one game and return a result dict.
    """
    import arc_agi as _arc_agi
    from object_extractor import ObjectExtractor
    from embedding_space import EmbeddingSpace
    from curiosity_explorer import CuriosityExplorer, MEANING_DIM
    from relational_transformer import RelationalTransformer, InteractionMemory, InteractionRecord, EFFECT_DIM
    from goal_planner import GoalInferenceModule, Planner
    from curiosity_explorer import compute_delta
    import torch
    import random

    arc = _arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)
    if env is None:
        return {"game_id": game_id, "error": "failed to create environment"}
    env.include_frame_data = True

    agent = ObjectCentricAgent(
        action_space=env.action_space,
        verbose=verbose,
    )

    recorder = None
    if record:
        from visualizer import Recorder
        recorder = Recorder(f"replays/{game_id}")

    obs = env.reset()
    grid = obs.frame[0]
    agent.reset(grid)

    step = 0
    levels_completed = 0
    t0 = time.time()
    max_total = max_steps_per_level * obs.win_levels

    while step < max_total:
        action, data = agent.step(grid, obs.state)

        if record and recorder and step % 5 == 0:
            goal = agent.goal_inf.best_goal()
            recorder.record(
                grid, agent.space, step,
                goal=goal,
                target_id=agent._prev_target_id,
                action_name=action.name,
            )

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

    elapsed = time.time() - t0

    if recorder:
        recorder.close()

    # Baseline: sum of per-level human baseline actions
    baseline = sum(env.environment_info.baseline_actions[:obs.win_levels])
    efficiency = levels_completed / max(step, 1)

    # Include current in-progress level stats (may not have been committed yet)
    all_stats = agent._level_stats + [agent._current_stats]
    total_explore = sum(s.exploration_steps for s in all_stats)
    total_plan = sum(s.planning_steps for s in all_stats)

    result = {
        "game_id": game_id,
        "steps": step,
        "levels_completed": levels_completed,
        "win_levels": obs.win_levels,
        "final_state": obs.state.name,
        "human_baseline": baseline,
        "efficiency_vs_human": baseline / max(step, 1) if levels_completed > 0 else 0.0,
        "explore_pct": total_explore / max(step, 1) * 100,
        "plan_pct": total_plan / max(step, 1) * 100,
        "priors_accumulated": len(agent._priors),
        "elapsed_s": round(elapsed, 1),
    }
    return result


# ------------------------------------------------------------------
# Multi-game evaluation
# ------------------------------------------------------------------

def print_table(results: List[dict]) -> None:
    header = (
        f"{'game_id':<22} {'steps':>6} {'lvl':>5} {'state':<10} "
        f"{'baseline':>9} {'eff%':>6} {'exp%':>6} {'plan%':>6} {'t(s)':>5}"
    )
    print()
    print(header)
    print("-" * len(header))

    total_steps = 0
    total_baseline = 0
    for r in results:
        if "error" in r:
            print(f"  {r['game_id']:<20} ERROR: {r['error']}")
            continue
        eff_pct = r["efficiency_vs_human"] * 100
        lvl_str = f"{r['levels_completed']}/{r['win_levels']}"
        print(
            f"  {r['game_id']:<20} {r['steps']:>6} {lvl_str:>5}  {r['final_state']:<10} "
            f"{r['human_baseline']:>9} {eff_pct:>5.1f}% "
            f"{r['explore_pct']:>5.1f}% {r['plan_pct']:>5.1f}% "
            f"{r['elapsed_s']:>5.1f}"
        )
        total_steps += r["steps"]
        total_baseline += r["human_baseline"]

    print("-" * len(header))
    overall_eff = total_baseline / max(total_steps, 1) * 100
    print(f"  {'TOTAL':<20} {total_steps:>6}  {'':>5}  {'':>10} "
          f"{total_baseline:>9} {overall_eff:>5.1f}%")
    print()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate ObjectCentricAgent on ARC-AGI-3")
    parser.add_argument("games", nargs="*", help="Game ID prefixes to run (default: --quick)")
    parser.add_argument("--quick", action="store_true", help="Run 3 quick games")
    parser.add_argument("--keyboard", action="store_true", help="Run keyboard-only games")
    parser.add_argument("--click", action="store_true", help="Run click games")
    parser.add_argument("--all", action="store_true", help="Run all available games")
    parser.add_argument("--record", action="store_true", help="Save replay frames")
    parser.add_argument("--steps", type=int, default=300, help="Max steps per level")
    parser.add_argument("--verbose", action="store_true", help="Verbose agent logging")
    args = parser.parse_args()

    # Resolve game list
    if args.games:
        # Look up full IDs for supplied prefixes
        arc = arc_agi.Arcade()
        all_envs = arc.get_environments()
        all_ids = [e.game_id for e in all_envs]
        game_ids = []
        for prefix in args.games:
            matches = [gid for gid in all_ids if gid.startswith(prefix)]
            if matches:
                game_ids.extend(matches)
            else:
                print(f"Warning: no game matching prefix '{prefix}'")
    elif args.all:
        arc = arc_agi.Arcade()
        game_ids = [e.game_id for e in arc.get_environments()]
    elif args.keyboard:
        game_ids = KEYBOARD_GAMES
    elif args.click:
        game_ids = CLICK_GAMES
    else:
        game_ids = QUICK_GAMES

    print(f"\nEvaluating {len(game_ids)} game(s)...")
    print(f"  max_steps_per_level={args.steps}  record={args.record}\n")

    results = []
    for gid in game_ids:
        print(f"  Running {gid}...", end="", flush=True)
        r = evaluate_game(
            gid,
            max_steps_per_level=args.steps,
            record=args.record,
            verbose=args.verbose,
        )
        status = r.get("final_state", r.get("error", "?"))
        print(f" {status} ({r.get('steps', '?')} steps)")
        results.append(r)

    print_table(results)


if __name__ == "__main__":
    main()
