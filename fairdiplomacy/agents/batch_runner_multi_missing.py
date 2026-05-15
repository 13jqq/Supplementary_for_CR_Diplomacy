import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


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


def agent_sig_for_csv(agents: List[str]) -> str:
    # Must match consistent_runner_multi.py: "-".join(sorted(set(assignment_map.values())))
    return "-".join(sorted(set(agents)))


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


def result_csv_path_for(root_dir: Path, agents: List[str], version: str, csv_name: Optional[str] = None) -> Path:
    if csv_name:
        return root_dir / csv_name
    return root_dir / f"results_setup=multi_agents={agent_sig_for_csv(agents)}_{version_lower(version)}.csv"


def generate_balanced_assignments(
    agents: List[str],
    block_seed: int,
    *,
    shuffle_powers: bool = True,
) -> List[Dict[str, str]]:
    """
    Same design as batch_runner_multi_random.py.

    For 4 agents and 7 powers, return 4 assignments.
    Inside one block_seed:
      - every power is assigned to every agent exactly once;
      - every agent controls 7 power-slots in total across the 4 games.
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


def iter_expected_tasks(args, agents: List[str], root_dir: Path):
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


def read_completed_from_csv(
    csv_path: Path,
    *,
    bad_end_reasons: Set[str],
) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], Dict[str, str]], List[Tuple[int, int]]]:
    """
    Returns:
      completed_keys: (block_seed, shift) rows that count as completed.
      row_by_key: last row seen for each key.
      duplicate_keys: keys observed more than once.

    If bad_end_reasons is non-empty, rows whose end_reason is in that set are treated as missing.
    """
    completed: Set[Tuple[int, int]] = set()
    row_by_key: Dict[Tuple[int, int], Dict[str, str]] = {}
    seen: Set[Tuple[int, int]] = set()
    duplicates: List[Tuple[int, int]] = []

    if not csv_path.exists():
        return completed, row_by_key, duplicates

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        needed = {"block_seed", "shift"}
        if not needed.issubset(fieldnames):
            raise ValueError(f"CSV must contain columns {needed}, got {reader.fieldnames}: {csv_path}")

        for row in reader:
            try:
                block_seed = int(str(row.get("block_seed", "")).strip())
                shift = int(str(row.get("shift", "")).strip())
            except Exception:
                continue

            key = (block_seed, shift)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
            row_by_key[key] = row

            end_reason = str(row.get("end_reason", "")).strip()
            if bad_end_reasons and end_reason in bad_end_reasons:
                continue
            completed.add(key)

    return completed, row_by_key, duplicates


def log_is_finished(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return ("=== RUN END ===" in text) and ("[CSV SAVED]" in text)


def print_assignment(task: Dict):
    print(
        f"[TASK] block_seed={task['block_seed']} shift={task['shift']} "
        f"game_seed={task['game_seed']} log={task['log_path'].name}"
    )
    for p in POWERS:
        print(f"  {p:<8} -> {task['assignment'][p]}")


def delete_log_if_needed(log_path: Path, *, dry_run: bool, delete_finished_logs: bool = False):
    if not log_path.exists():
        return

    if log_is_finished(log_path) and not delete_finished_logs:
        print(f"[KEEP FINISHED LOG] {log_path}")
        return

    print(f"[DELETE LOG] {log_path}")
    if not dry_run:
        log_path.unlink()


def run_one(task: Dict, args, root_dir: Path) -> int:
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


def write_missing_plan(root_dir: Path, version: str, missing_tasks: List[Dict]) -> Path:
    path = root_dir / f"missing_plan_{version_lower(version)}.jsonl"
    root_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in missing_tasks:
            rec = {
                "block_seed": task["block_seed"],
                "shift": task["shift"],
                "game_seed": task["game_seed"],
                "assignment": task["assignment"],
                "log_path": str(task["log_path"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def summarize_completed_balance(
    expected_tasks: List[Dict],
    completed_keys: Set[Tuple[int, int]],
):
    counts = {agent: {p: 0 for p in POWERS} for task in expected_tasks for agent in set(task["assignment"].values())}
    total_slots = {agent: 0 for agent in counts}

    for task in expected_tasks:
        key = (task["block_seed"], task["shift"])
        if key not in completed_keys:
            continue
        for p, agent in task["assignment"].items():
            counts[agent][p] += 1
            total_slots[agent] += 1

    print("[COMPLETED BALANCE CHECK]")
    for agent in sorted(counts):
        vals = " ".join(f"{p[:3]}={counts[agent][p]}" for p in POWERS)
        print(f"  {agent:<28} total_slots={total_slots[agent]:<4} {vals}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find and rerun missing jobs for batch_runner_multi_random.py. "
            "Completion is judged by rows in the multi-agent result CSV, not merely by log existence."
        )
    )

    parser.add_argument(
        "--agents",
        type=str,
        required=True,
        help="Exactly 4 comma-separated agent kinds, same as batch_runner_multi_random.py.",
    )

    parser.add_argument("--version", default="V1")
    parser.add_argument("--source", default="bqre_topK", choices=["bqre_topK", "search_br", "bp"])
    parser.add_argument("--mode", default="bqre", choices=["top1", "sample", "bqre", "constrained_bqre"])
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max_phases", type=int, default=60)

    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=9)
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
        help="Optional explicit root directory for logs/results. Must match the original batch run if used.",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default=None,
        help="Optional exact CSV filename under root_dir. Normally not needed.",
    )
    parser.add_argument(
        "--no_shuffle_powers",
        action="store_true",
        help="Must match the original batch run if you used --no_shuffle_powers there.",
    )

    parser.add_argument(
        "--bad_end_reasons",
        type=str,
        default="",
        help=(
            "Comma-separated end_reason values to treat as missing and rerun, e.g. "
            "exception,stopped_unknown. By default, any CSV row counts as completed."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print missing jobs / commands; do not delete logs or execute reruns.",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Print missing jobs and balance check, but do not delete logs or rerun.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if a rerun command returns non-zero.",
    )
    parser.add_argument(
        "--delete_finished_logs",
        action="store_true",
        help="Dangerous: also delete finished logs for rows treated as missing, e.g. bad_end_reason reruns.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    agents = parse_agents(args.agents)
    root_dir = build_root_dir(args, agents)
    csv_path = result_csv_path_for(root_dir, agents, args.version, args.csv_name)
    bad_end_reasons = {x.strip() for x in str(args.bad_end_reasons).split(",") if x.strip()}

    expected_tasks = list(iter_expected_tasks(args, agents, root_dir))
    expected_keys = {(t["block_seed"], t["shift"]) for t in expected_tasks}

    completed_keys, row_by_key, duplicate_keys = read_completed_from_csv(
        csv_path,
        bad_end_reasons=bad_end_reasons,
    )
    completed_keys &= expected_keys

    missing_tasks = [
        t for t in expected_tasks
        if (t["block_seed"], t["shift"]) not in completed_keys
    ]

    print(f"[ROOT_DIR] {root_dir}")
    print(f"[CSV] {csv_path}")
    print(f"[AGENTS] {agents}")
    print(f"[SEEDS] {args.seed_start}..{args.seed_end}")
    print(f"[EXPECTED] {len(expected_tasks)} jobs = {(args.seed_end - args.seed_start + 1)} block_seeds * 4 shifts")
    print(f"[CSV_ROWS_IN_RANGE_COUNTED_COMPLETED] {len(completed_keys)}")
    print(f"[MISSING] {len(missing_tasks)}")

    if bad_end_reasons:
        print(f"[BAD_END_REASONS_TREATED_AS_MISSING] {sorted(bad_end_reasons)}")

    if duplicate_keys:
        uniq_dups = sorted(set(duplicate_keys))
        print(f"[WARN] duplicate rows in CSV for {len(uniq_dups)} task keys, examples={uniq_dups[:10]}")

    summarize_completed_balance(expected_tasks, completed_keys)

    if not missing_tasks:
        print("[DONE] no missing jobs found")
        return

    missing_plan = write_missing_plan(root_dir, args.version, missing_tasks)
    print(f"[MISSING_PLAN] {missing_plan}")

    print("[MISSING TASKS]")
    for task in missing_tasks:
        key = (task["block_seed"], task["shift"])
        row = row_by_key.get(key)
        reason = "no_csv_row"
        if row is not None:
            reason = f"bad_end_reason={row.get('end_reason', '')}"
        print(
            f"  block_seed={task['block_seed']} shift={task['shift']} "
            f"game_seed={task['game_seed']} reason={reason} log={task['log_path']}"
        )
        if args.dry_run or args.report_only:
            for p in POWERS:
                print(f"    {p:<8} -> {task['assignment'][p]}")

    if args.report_only:
        print("[REPORT_ONLY] no logs deleted and no commands executed")
        return

    # Delete stale claimed/crashed logs so reruns can write clean logs.
    for task in missing_tasks:
        delete_log_if_needed(
            task["log_path"],
            dry_run=args.dry_run,
            delete_finished_logs=args.delete_finished_logs,
        )

    failed = []
    for task in missing_tasks:
        rc = run_one(task, args, root_dir)
        if rc != 0:
            failed.append((task["block_seed"], task["shift"], task["game_seed"], rc))
            print(
                f"[FAILED] block_seed={task['block_seed']} shift={task['shift']} "
                f"game_seed={task['game_seed']} returncode={rc}"
            )
            if args.stop_on_error:
                sys.exit(rc)
        else:
            print(
                f"[OK] block_seed={task['block_seed']} shift={task['shift']} "
                f"game_seed={task['game_seed']}"
            )

    if failed:
        print("[FINAL] some reruns failed:")
        for block_seed, shift, game_seed, rc in failed:
            print(f"  - block_seed={block_seed}, shift={shift}, game_seed={game_seed}, returncode={rc}")
        sys.exit(1)

    print("[FINAL] all missing jobs completed")


if __name__ == "__main__":
    main()
