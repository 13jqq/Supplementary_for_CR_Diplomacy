# coding=utf-8
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple

from consistent_agent_V3 import POWERS, get_territory_parts, load_consistent_agent


ActionInfo = Dict[str, Any]
JointOrders = Dict[str, List[str]]


def parse_args() -> argparse.Namespace:
    """
    Input:
      - command-line configuration for a CR^2 evaluation run.

    Output:
      - parsed runtime arguments.
    """
    parser = argparse.ArgumentParser(description="Run CR^2 agents in a dipcc game and log candidate-level diagnostics.")

    parser.add_argument("--cfg", type=str, default="conf/common/agents/consistent_agent.prototxt")
    parser.add_argument("--project_root", type=str, default="/workspace/Diplomacy/diplomacy_cicero")
    parser.add_argument("--power", type=str, default="AUSTRIA", choices=POWERS)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--source", type=str, default="bqre_topK", choices=["bqre_topK", "search_br", "bp"])
    parser.add_argument("--mode", type=str, default="top1", choices=["top1", "sample", "bqre"])
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max_phases", type=int, default=60)

    parser.add_argument("--log_dir", type=str, default="logs_consistent")
    parser.add_argument("--log", type=str, default=None)

    parser.add_argument("--log_candidates", action="store_true")
    parser.add_argument("--max_logged_candidates", type=int, default=10)

    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """
    Input:
      - seed: integer random seed.

    Output:
      - None. The function initializes major pseudo-random generators used by
        the policy/search pipeline.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def enter_project_root(project_root: str | None) -> None:
    """
    Input:
      - project_root: repository root used for relative model and config paths.

    Output:
      - None. The current working directory is changed when the path exists.
    """
    if project_root and os.path.exists(project_root):
        os.chdir(project_root)


def build_log_path(args: argparse.Namespace) -> str:
    """
    Input:
      - args.log: explicit log path, if provided.
      - args.log_dir: directory for generated logs.

    Output:
      - concrete log file path.
    """
    if args.log:
        return args.log

    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(os.getcwd(), args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"cr2_run_{timestamp}.log")


def is_game_finished(game: Any) -> bool:
    """
    Input:
      - game: active pydipcc game.

    Output:
      - True if the game has reached a terminal phase, otherwise False.
    """
    for attr in ("is_game_done", "is_game_over", "game_over"):
        if hasattr(game, attr):
            try:
                value = getattr(game, attr)
                return bool(value() if callable(value) else value)
            except Exception:
                pass

    phase = str(game.get_current_phase()).upper()
    return ("COMPLETED" in phase) or (phase in {"DONE", "END"})


def read_state_dict(game: Any) -> Dict[str, Any]:
    """
    Input:
      - game: active pydipcc game.

    Output:
      - state dictionary suitable for territory and unit diagnostics.
    """
    try:
        state = game.get_state()
    except Exception:
        return {}

    if not isinstance(state, dict) and hasattr(state, "to_dict"):
        try:
            state = state.to_dict()
        except Exception:
            return {}

    return state if isinstance(state, dict) else {}


def initialize_agent_states(agent: Any) -> Dict[str, Any]:
    """
    Input:
      - agent: CR^2-compatible agent.

    Output:
      - per-power recurrent state objects used by the underlying policy/search
        model.
    """
    return {power: agent.initialize_state(power) for power in POWERS}


def action_to_string(action: Any) -> str:
    """
    Input:
      - action: tuple/list/string representation of unit-level orders.

    Output:
      - compact string representation for log files.
    """
    if isinstance(action, (list, tuple)):
        return "[" + ", ".join(map(str, action)) + "]"
    return str(action)


def write_run_header(handle: Any, args: argparse.Namespace, log_path: str) -> None:
    """
    Input:
      - handle: writable log stream.
      - args: runtime configuration.
      - log_path: generated log path.

    Output:
      - None. The function writes the static run configuration.
    """
    handle.write("=== CR2 RUN START ===\n")
    handle.write(f"cwd={os.getcwd()}\n")
    handle.write(f"cfg={args.cfg}\n")
    handle.write(f"log_path={log_path}\n")
    handle.write(
        f"power={args.power}, seed={args.seed}, source={args.source}, "
        f"mode={args.mode}, topk={args.topk}, max_phases={args.max_phases}\n\n"
    )
    handle.flush()


def write_state_summary(handle: Any, state: Dict[str, Any]) -> None:
    """
    Input:
      - handle: writable log stream.
      - state: current game state dictionary.

    Output:
      - None. The function records units, supply centers, and territory traces
        used by the structural quality terms.
    """
    units = state.get("units", {}) or {}
    influence = state.get("influence", None)
    territory_source = "influence" if isinstance(influence, dict) else "state_units_and_centers"

    handle.write(f"[STATE BEFORE] territory_source={territory_source}\n")
    for power in POWERS:
        unit_list = list((units.get(power) or []))
        sc_set, unit_set, past_free_set = get_territory_parts(state, power)
        territory_set = sc_set | unit_set | past_free_set

        sc_list = sorted(sc_set)
        non_sc_list = sorted(territory_set - sc_set)

        handle.write(
            f"  {power}: "
            f"units({len(unit_list)})={unit_list} | "
            f"SC({len(sc_list)})={sc_list} | "
            f"nonSC({len(non_sc_list)})={non_sc_list}\n"
        )


def collect_cr2_decisions(
    agent: Any,
    game: Any,
    states: Dict[str, Any],
    *,
    source: str,
    top_k: int,
    mode: str,
) -> Tuple[JointOrders, Dict[str, ActionInfo]]:
    """
    Input:
      - agent: CR^2-compatible decision agent.
      - game: current pydipcc game.
      - states: per-power agent states.
      - source/top_k/mode: candidate-generation and action-selection settings.

    Output:
      - joint_orders: final order list for each power.
      - decision_info: per-power candidate distribution and structural
        diagnostics returned by the agent.
    """
    joint_orders: JointOrders = {power: [] for power in POWERS}
    decision_info: Dict[str, ActionInfo] = {}

    for power in POWERS:
        info = agent.get_orders_info(
            game=game,
            power=power,
            state=states[power],
            source=source,
            top_k=top_k,
            mode=mode,
        )

        orders = list(info.get("orders", []) or [])
        joint_orders[power] = orders
        decision_info[power] = {
            "orders": orders,
            "items": info.get("items", []) or [],
            "raw_items": info.get("raw_items", []) or [],
            "used_source": info.get("used_source", source),
            "structural_metrics": info.get("structural_metrics", []) or [],
        }

    return joint_orders, decision_info


def write_candidate_diagnostics(
    handle: Any,
    decision_info: Dict[str, ActionInfo],
    *,
    log_candidates: bool,
    max_logged_candidates: int,
) -> None:
    """
    Input:
      - handle: writable log stream.
      - decision_info: per-power CR^2 output.
      - log_candidates: whether to print candidate-level details.
      - max_logged_candidates: maximum number of candidates written per power.

    Output:
      - None. The function records the structure-regularized candidate
        distribution rather than hard-removal events.
    """
    for power in POWERS:
        info = decision_info[power]
        items = info["items"]
        raw_items = info["raw_items"]
        metrics = info["structural_metrics"]

        handle.write(
            f"[AGENT] power={power} used_source={info['used_source']} "
            f"raw_candidates={len(raw_items)} regularized_candidates={len(items)}\n"
        )
        handle.write(f"[SELECTED] power={power} orders={info['orders']}\n")

        if not log_candidates:
            continue

        n = min(max_logged_candidates, len(metrics))
        handle.write(f"[STRUCTURAL SCORES] power={power} n={n}\n")

        for idx, metric in enumerate(metrics[:n]):
            action = metric.get("action", ())
            prob = float(metric.get("regularized_prob", 0.0))
            u_value = float(metric.get("u", 0.0))
            phi = int(metric.get("phi", 0))
            psi = int(metric.get("psi", 0))
            penalty = float(metric.get("structural_penalty", 0.0))
            augmented_u = float(metric.get("augmented_u", u_value))

            handle.write(
                f"  #{idx:02d} prob={prob:.8f} "
                f"u={u_value:.8f} phi={phi} psi={psi} "
                f"penalty={penalty:.8f} augmented_u={augmented_u:.8f} "
                f"action={action_to_string(action)}\n"
            )


def submit_joint_orders(game: Any, joint_orders: JointOrders) -> None:
    """
    Input:
      - game: active pydipcc game.
      - joint_orders: final orders for all powers.

    Output:
      - None. The function submits all orders only after every power has
        completed its decision step, preserving simultaneous-action semantics.
    """
    for power in POWERS:
        game.set_orders(power, joint_orders.get(power, []))


def write_joint_orders(handle: Any, joint_orders: JointOrders) -> None:
    """
    Input:
      - handle: writable log stream.
      - joint_orders: final submitted orders.

    Output:
      - None. The function writes the executed joint action.
    """
    handle.write("[ORDERS SET]\n")
    for power in POWERS:
        handle.write(f"  {power}: {joint_orders.get(power, [])}\n")


def run_cr2_game(args: argparse.Namespace) -> str:
    """
    Input:
      - args: runtime configuration.

    Output:
      - path to the generated log file.
    """
    from fairdiplomacy import pydipcc

    set_global_seed(args.seed)
    enter_project_root(args.project_root)

    log_path = build_log_path(args)
    agent = load_consistent_agent(args.cfg, skip_cache=False)
    game = pydipcc.Game()
    states = initialize_agent_states(agent)

    with open(log_path, "w", encoding="utf-8") as handle:
        write_run_header(handle, args, log_path)

        step = 0
        while step < args.max_phases and not is_game_finished(game):
            phase = game.get_current_phase()
            state = read_state_dict(game)

            joint_orders, decision_info = collect_cr2_decisions(
                agent,
                game,
                states,
                source=args.source,
                top_k=args.topk,
                mode=args.mode,
            )

            handle.write("\n" + "=" * 90 + "\n")
            handle.write(f"[STEP {step:04d}] phase={phase}\n")
            write_state_summary(handle, state)
            write_candidate_diagnostics(
                handle,
                decision_info,
                log_candidates=args.log_candidates,
                max_logged_candidates=args.max_logged_candidates,
            )

            submit_joint_orders(game, joint_orders)
            write_joint_orders(handle, joint_orders)

            try:
                game.process()
            except Exception as exc:
                handle.write(f"[ERROR] game.process() failed @phase={phase}: {repr(exc)}\n")
                break

            handle.flush()
            step += 1

        handle.write("\n=== RUN END ===\n")
        handle.write(f"final_phase={game.get_current_phase()}\n")
        handle.flush()

    return log_path


def main() -> None:
    """
    Input:
      - command-line arguments.

    Output:
      - None. The generated log path is printed to stdout.
    """
    args = parse_args()
    log_path = run_cr2_game(args)
    print(f"[OK] log saved to: {log_path}")


if __name__ == "__main__":
    main()
