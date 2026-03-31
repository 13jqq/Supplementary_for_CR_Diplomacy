import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Set


POWERS = [
    "AUSTRIA",
    "ENGLAND",
    "FRANCE",
    "GERMANY",
    "ITALY",
    "RUSSIA",
    "TURKEY",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT_DIRS = {
    "dipnet": PROJECT_ROOT / "logs_batch" / "log_dipnet_V1",
    "searchbot": PROJECT_ROOT / "logs_batch" / "log_searchbot_V1",
    "diplodocus_high": PROJECT_ROOT / "logs_batch" / "log_diplodocus_high_V1",
    "cicero_nopress": PROJECT_ROOT / "logs_batch" / "log_cicero_nopress_V1",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan per-power CSVs, find missing seeds, delete incomplete logs, and rerun missing jobs."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dipnet", action="store_true")
    group.add_argument("--searchbot", action="store_true")
    group.add_argument("--diplodocus_high", action="store_true")
    group.add_argument("--cicero_nopress", action="store_true")

    parser.add_argument("--setup", default="1v6")
    parser.add_argument("--my_agent", default="consistent")
    parser.add_argument("--source", default="bqre_topK")
    parser.add_argument("--mode", default="top1")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=9)

    # 可选项
    parser.add_argument(
        "--csv_name",
        type=str,
        default=None,
        help="If provided, use this exact csv filename inside each power directory.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print missing seeds / files to delete / commands, do not execute.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if any rerun command returns non-zero.",
    )

    return parser.parse_args()


def resolve_opp_agent(args):
    if args.dipnet:
        return "dipnet"
    if args.searchbot:
        return "searchbot"
    if args.diplodocus_high:
        return "diplodocus_high"
    if args.cicero_nopress:
        return "cicero_nopress"
    raise ValueError("No opponent agent selected")


def ensure_dirs(root_dir: Path):
    root_dir.mkdir(parents=True, exist_ok=True)
    for power in POWERS:
        (root_dir / power).mkdir(parents=True, exist_ok=True)


def log_path_for(root_dir: Path, power: str, opp_agent: str, seed: int) -> Path:
    return root_dir / power / f"run_1v6_my{power}_myconsistent_opp{opp_agent}_seed{seed}.log"


def find_seed_csv(power_dir: Path, csv_name: Optional[str] = None) -> Optional[Path]:
    """
    Find a csv file containing a 'seed' column.
    Preference:
      1) exact csv_name if provided and valid
      2) among *.csv files, choose the newest one that contains 'seed'
    """
    if csv_name is not None:
        p = power_dir / csv_name
        if p.exists() and p.is_file():
            if csv_has_seed_column(p):
                return p
            raise ValueError(f"CSV exists but has no 'seed' column: {p}")
        return None

    candidates = sorted(power_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if csv_has_seed_column(p):
            return p
    return None


def csv_has_seed_column(csv_path: Path) -> bool:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if not header:
            return False
        return "seed" in [h.strip() for h in header]
    except Exception:
        return False


def read_completed_seeds(csv_path: Path, seed_start: int, seed_end: int) -> Set[int]:
    completed = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "seed" not in (reader.fieldnames or []):
            raise ValueError(f"'seed' column not found in {csv_path}")
        for row in reader:
            raw = row.get("seed", "")
            try:
                seed = int(str(raw).strip())
            except Exception:
                continue
            if seed_start <= seed <= seed_end:
                completed.add(seed)
    return completed


def compute_missing_seeds(csv_path: Optional[Path], seed_start: int, seed_end: int) -> List[int]:
    target = set(range(seed_start, seed_end + 1))
    if csv_path is None:
        return sorted(target)

    completed = read_completed_seeds(csv_path, seed_start, seed_end)
    missing = sorted(target - completed)
    return missing


def delete_incomplete_log_if_exists(log_path: Path, dry_run: bool = False):
    if log_path.exists():
        print(f"[DELETE LOG] {log_path}")
        if not dry_run:
            log_path.unlink()


def run_one(power: str, seed: int, opp_agent: str, root_dir: Path, args) -> int:
    log_path = log_path_for(root_dir, power, opp_agent, seed)

    cmd = [
        sys.executable,
        "-m",
        "consistent_runner_for",
        "--setup",
        args.setup,
        "--power",
        power,
        "--seed",
        str(seed),
        "--my_agent",
        args.my_agent,
        "--opp_agent",
        opp_agent,
        "--source",
        args.source,
        "--mode",
        args.mode,
        "--log_dir",
        str(root_dir / power),
        "--log",
        str(log_path),
    ]

    print(f"[RUN] power={power} seed={seed} opp={opp_agent}")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    result = subprocess.run(cmd)
    return result.returncode


def main():
    args = parse_args()
    opp_agent = resolve_opp_agent(args)
    root_dir = ROOT_DIRS[opp_agent]

    ensure_dirs(root_dir)

    total_missing = 0
    plan = []

    # 先扫描
    print(f"[SCAN] opponent={opp_agent}")
    for power in POWERS:
        power_dir = root_dir / power
        csv_path = find_seed_csv(power_dir, args.csv_name)
        missing = compute_missing_seeds(csv_path, args.seed_start, args.seed_end)

        if csv_path is None:
            print(f"[WARN] {power}: no csv with 'seed' column found under {power_dir}, treat all as missing")
        else:
            print(f"[CSV]  {power}: {csv_path}")

        print(f"[MISS] {power}: {missing}")

        for seed in missing:
            plan.append((power, seed))
        total_missing += len(missing)

    print(f"[SUMMARY] total missing jobs = {total_missing}")

    if total_missing == 0:
        print("[DONE] no missing seeds found")
        return

    # 删除旧 log
    for power, seed in plan:
        log_path = log_path_for(root_dir, power, opp_agent, seed)
        delete_incomplete_log_if_exists(log_path, dry_run=args.dry_run)

    # 补跑
    failed = []
    for power, seed in plan:
        rc = run_one(power, seed, opp_agent, root_dir, args)
        if rc != 0:
            failed.append((power, seed, rc))
            print(f"[FAILED] power={power} seed={seed} returncode={rc}")
            if args.stop_on_error:
                sys.exit(rc)
        else:
            print(f"[OK] power={power} seed={seed}")

    if failed:
        print("[FINAL] some reruns failed:")
        for power, seed, rc in failed:
            print(f"  - power={power}, seed={seed}, returncode={rc}")
        sys.exit(1)

    print("[FINAL] all missing jobs completed")


if __name__ == "__main__":
    main()