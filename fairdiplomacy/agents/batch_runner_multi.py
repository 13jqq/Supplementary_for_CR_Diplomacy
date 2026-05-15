import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


POWERS = [
    "AUSTRIA",
    "ENGLAND",
    "FRANCE",
    "GERMANY",
    "ITALY",
    "RUSSIA",
    "TURKEY",
]

AGENT_CHOICES = [
    "consistent",
    "consistent_docus",
    "cicero_nopress",
    "diplodocus_high",
    "diplodocus_low",
    "searchbot",
    "dipnet",
    "searchbot_neurips21_dora",
]

AGENT_FOLDER_ALIAS = {
    "consistent": "consistent",
    "consistent_docus": "consistent_docus",
    "cicero_nopress": "cicero",
    "diplodocus_high": "diplodocus_high",
    "diplodocus_low": "diplodocus_low",
    "searchbot": "searchbot",
    "dipnet": "dipnet",
    "searchbot_neurips21_dora": "dora",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def normalize_version_tag(version: str) -> str:
    s = str(version).strip()
    if not s:
        return "V1"
    if s[0] in ("v", "V"):
        s = s[1:]
    return f"V{s}"


def version_lower(version: str) -> str:
    return normalize_version_tag(version).lower()


def parse_agents(s: str) -> List[str]:
    agents = [x.strip() for x in str(s).split(",") if x.strip()]
    if len(agents) != 4:
        raise ValueError(
            f"--agents must contain exactly 4 comma-separated agents, got {len(agents)}: {agents}"
        )
    bad = [a for a in agents if a not in AGENT_CHOICES]
    if bad:
        raise ValueError(f"Unknown agent(s): {bad}; choices={AGENT_CHOICES}")
    if len(set(agents)) != len(agents):
        raise ValueError(f"--agents contains duplicates: {agents}")
    return agents


def agents_slug(agents: List[str]) -> str:
    return "__".join(AGENT_FOLDER_ALIAS.get(a, a) for a in agents)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Balanced randomized multi-agent Diplomacy batch runner. "
            "For 4 agents and 7 powers, each block seed runs 4 games; "
            "within each block, every agent controls every power exactly once."
        )
    )

    parser.add_argument(
        "--agents",
        type=str,
        required=True,
        help=(
            "Exactly 4 comma-separated agent kinds, e.g. "
            "consistent,consistent_docus,cicero_nopress,diplodocus_high"
        ),
    )

    parser.add_argument("--version", default="V1")
    parser.add_argument("--source", default="bqre_topK", choices=["bqre_topK", "search_br", "bp"])
    parser.add_argument("--mode", default="bqre", choices=["top1", "sample", "bqre", "constrained_bqre"])
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max_phases", type=int, default=60)

    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=9)

    # game_seed = block_seed * seed_multiplier + shift
    parser.add_argument("--seed_multiplier", type=int, default=1000)

    parser.add_argument(
        "--runner_module",
        type=str,
        default="fairdiplomacy.agents.consistent_runner_multi",
        help="Module path of the single-game runner.",
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Project root passed to the single-game runner.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default=None,
        help="Optional explicit root directory for logs. Default: PROJECT_ROOT/logs_batch/multi_random/...",
    )

    parser.add_argument(
        "--no_shuffle_powers",
        action="store_true",
        help="Disable shuffling of power order inside each block. Normally keep this off.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print generated commands without executing them.",
    )
    parser.add_argument(
        "--list_assignments",
        action="store_true",
        help="Only print all assignments and exit.",
    )

    return parser.parse_args()


def generate_balanced_assignments(
    agents: List[str],
    block_seed: int,
    *,
    shuffle_powers: bool = True,
) -> List[Dict[str, str]]:
    """
    For 4 agents and 7 powers, return 4 assignments.

    Property inside one block:
      - every power is assigned to every agent exactly once;
      - every agent controls 7 power-slots in total across the 4 games;
      - in each single game, three agents control 2 powers and one agent controls 1 power.
    """
    if len(agents) != 4:
        raise ValueError("This runner is designed for exactly 4 agents.")

    rng = random.Random(block_seed)

    shuffled_agents = agents[:]
    rng.shuffle(shuffled_agents)

    powers = POWERS[:]
    if shuffle_powers:
        rng.shuffle(powers)

    assignments: List[Dict[str, str]] = []
    n = len(shuffled_agents)

    for shift in range(n):
        assignment = {}
        for i, power in enumerate(powers):
            assignment[power] = shuffled_agents[(i + shift) % n]
        assignments.append(assignment)

    return assignments


def build_root_dir(args, agents: List[str]) -> Path:
    version_tag = normalize_version_tag(args.version)
    if args.log_root:
        return Path(args.log_root).resolve()

    return (
        Path(args.project_root).resolve()
        / "logs_batch"
        / "multi_random"
        / agents_slug(agents)
        / f"log_multi_random_{agents_slug(agents)}_{version_tag}"
    )


def log_path_for(root_dir: Path, block_seed: int, shift: int, game_seed: int, version: str) -> Path:
    pdir = root_dir / f"block_seed_{block_seed:04d}"
    return pdir / f"run_multi_block{block_seed}_shift{shift}_seed{game_seed}_{version_lower(version)}.log"


def try_claim_log(log_path: Path, payload: Dict) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd = os.open(str(log_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("[CLAIMED BY BATCH_RUNNER_MULTI_RANDOM]\n")
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    return True


def iter_tasks(args, agents: List[str], root_dir: Path):
    for block_seed in range(args.seed_start, args.seed_end + 1):
        assignments = generate_balanced_assignments(
            agents,
            block_seed,
            shuffle_powers=(not args.no_shuffle_powers),
        )

        for shift, assignment in enumerate(assignments):
            game_seed = block_seed * args.seed_multiplier + shift
            log_path = log_path_for(root_dir, block_seed, shift, game_seed, args.version)

            yield {
                "block_seed": block_seed,
                "shift": shift,
                "game_seed": game_seed,
                "assignment": assignment,
                "log_path": log_path,
            }


def find_and_claim_next_task(args, agents: List[str], root_dir: Path):
    for task in iter_tasks(args, agents, root_dir):
        log_path = task["log_path"]
        if log_path.exists():
            continue

        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "block_seed": task["block_seed"],
            "shift": task["shift"],
            "game_seed": task["game_seed"],
            "agents": agents,
            "assignment": task["assignment"],
            "version": normalize_version_tag(args.version),
            "source": args.source,
            "mode": args.mode,
            "topk": args.topk,
            "max_phases": args.max_phases,
            "runner_module": args.runner_module,
            "log_path": str(log_path),
        }

        if try_claim_log(log_path, payload):
            return task

    return None


def write_assignment_plan(args, agents: List[str], root_dir: Path):
    root_dir.mkdir(parents=True, exist_ok=True)
    plan_path = root_dir / f"assignment_plan_{version_lower(args.version)}.jsonl"

    with open(plan_path, "w", encoding="utf-8") as f:
        for task in iter_tasks(args, agents, root_dir):
            rec = {
                "block_seed": task["block_seed"],
                "shift": task["shift"],
                "game_seed": task["game_seed"],
                "assignment": task["assignment"],
                "log_path": str(task["log_path"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    return plan_path


def print_assignment(task: Dict):
    print(
        f"[TASK] block_seed={task['block_seed']} shift={task['shift']} "
        f"game_seed={task['game_seed']}"
    )
    for p in POWERS:
        print(f"  {p:<8} -> {task['assignment'][p]}")


def run_one_task(task: Dict, args, root_dir: Path) -> int:
    log_path = task["log_path"]
    assignment_json = json.dumps(task["assignment"], ensure_ascii=False, sort_keys=True)

    cmd = [
        sys.executable,
        "-m",
        args.runner_module,
        "--setup",
        "multi",
        "--seed",
        str(task["game_seed"]),
        "--block_seed",
        str(task["block_seed"]),
        "--shift",
        str(task["shift"]),
        "--assignment_json",
        assignment_json,
        "--source",
        args.source,
        "--mode",
        args.mode,
        "--topk",
        str(args.topk),
        "--max_phases",
        str(args.max_phases),
        "--exp_version",
        normalize_version_tag(args.version),
        "--project_root",
        args.project_root,
        "--log_dir",
        str(root_dir),
        "--log",
        str(log_path),
    ]

    print_assignment(task)
    print("[CMD]", " ".join(cmd))

    if args.dry_run:
        return 0

    result = subprocess.run(cmd)
    return result.returncode


def main():
    args = parse_args()
    agents = parse_agents(args.agents)
    root_dir = build_root_dir(args, agents)

    print(f"[ROOT_DIR] {root_dir}")
    print(f"[AGENTS] {agents}")
    print(f"[SEEDS] {args.seed_start}..{args.seed_end}")
    print("[DESIGN] each block_seed runs 4 shifts; each agent plays each power exactly once per block.")

    plan_path = write_assignment_plan(args, agents, root_dir)
    print(f"[PLAN] {plan_path}")

    if args.list_assignments:
        for task in iter_tasks(args, agents, root_dir):
            print_assignment(task)
        return

    while True:
        task = find_and_claim_next_task(args, agents, root_dir)

        if task is None:
            print("[DONE] no remaining tasks.")
            break

        try:
            rc = run_one_task(task, args, root_dir)

            if rc != 0:
                with open(task["log_path"], "a", encoding="utf-8") as f:
                    f.write(f"\n[BATCH_RUNNER_MULTI_RANDOM_ERROR] returncode={rc}\n")
                print(
                    f"[FAILED] block_seed={task['block_seed']} shift={task['shift']} "
                    f"game_seed={task['game_seed']} returncode={rc}"
                )
                sys.exit(rc)

            print(
                f"[OK] block_seed={task['block_seed']} shift={task['shift']} "
                f"game_seed={task['game_seed']}"
            )

        except KeyboardInterrupt:
            print(
                f"[STOPPED] block_seed={task['block_seed']} shift={task['shift']} "
                f"game_seed={task['game_seed']}"
            )
            raise
        except Exception as e:
            with open(task["log_path"], "a", encoding="utf-8") as f:
                f.write(f"\n[BATCH_RUNNER_MULTI_RANDOM_EXCEPTION] {repr(e)}\n")
            print(f"[EXCEPTION] {e}")
            raise


if __name__ == "__main__":
    main()
