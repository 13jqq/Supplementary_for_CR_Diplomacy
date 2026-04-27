import argparse
import csv
import subprocess
import sys
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 仅用于文件夹显示名；csv/log 和命令行里仍保留真实 agent 名称
AGENT_FOLDER_ALIAS = {
    "consistent": "consistent",
    "consistent_docus": "consistent_docus",
    "cicero_nopress": "cicero",
    "diplodocus_high": "diplodocus",
    "diplodocus_low": "diplodocus_low",
    "searchbot": "searchbot",
    "dipnet": "dipnet",
    "searchbot_neurips21_dora": "dora",
}


def normalize_version_tag(version: str) -> str:
    s = str(version).strip()
    if not s:
        return "V1"
    if s[0] in ("v", "V"):
        s = s[1:]
    return f"V{s}"


def version_lower(version: str) -> str:
    return normalize_version_tag(version).lower()


def power_dir_name(power: str) -> str:
    return str(power).strip().title()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan per-power CSVs, find missing seeds, delete incomplete logs, and rerun missing jobs under the new generic batch-runner layout."
    )

    parser.add_argument("--setup", default="1v6", choices=["1v6", "all7"])
    parser.add_argument("--my_agent", default="consistent", choices=AGENT_CHOICES)
    parser.add_argument("--opp_agent", required=True, choices=AGENT_CHOICES)
    parser.add_argument("--all_agent", default="consistent", choices=AGENT_CHOICES)

    parser.add_argument("--version", default="V1")
    parser.add_argument("--source", default="bqre_topK")
    parser.add_argument("--mode", default="bqre")
    parser.add_argument("--topk", type=int, default=30)

    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=9)

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


def build_root_dir(args) -> Path:
    version_tag = normalize_version_tag(args.version)
    my_dir = f"{args.my_agent}_{args.setup}"

    my_disp = AGENT_FOLDER_ALIAS.get(args.my_agent, args.my_agent)
    opp_disp = AGENT_FOLDER_ALIAS.get(args.opp_agent, args.opp_agent)

    return PROJECT_ROOT / "logs_batch" / my_dir / f"log_{my_disp}_vs_{opp_disp}_{version_tag}"


def ensure_dirs(root_dir: Path):
    root_dir.mkdir(parents=True, exist_ok=True)
    for power in POWERS:
        (root_dir / power_dir_name(power)).mkdir(parents=True, exist_ok=True)


def log_path_for(root_dir: Path, power: str, my_agent: str, opp_agent: str, seed: int, version: str) -> Path:
    pdir = root_dir / power_dir_name(power)
    vlow = version_lower(version)
    return pdir / f"run_1v6_my{power}_my{my_agent}_opp{opp_agent}_seed{seed}_{vlow}.log"


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


def read_completed_seeds(csv_path: Path, seed_start: int, seed_end: int) -> Set[int]:
    completed = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [x.strip() for x in (reader.fieldnames or [])]
        if "seed" not in fieldnames:
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


def run_one(power: str, seed: int, root_dir: Path, args) -> int:
    log_path = log_path_for(
        root_dir=root_dir,
        power=power,
        my_agent=args.my_agent,
        opp_agent=args.opp_agent,
        seed=seed,
        version=args.version,
    )
    power_dir = root_dir / power_dir_name(power)

    cmd = [
        sys.executable,
        "-m",
        "fairdiplomacy.agents.consistent_runner_for",
        "--setup", args.setup,
        "--power", power,
        "--seed", str(seed),
        "--my_agent", args.my_agent,
        "--opp_agent", args.opp_agent,
        "--all_agent", args.all_agent,
        "--source", args.source,
        "--mode", args.mode,
        "--topk", str(args.topk),
        "--exp_version", normalize_version_tag(args.version),
        "--log_dir", str(power_dir),
        "--log", str(log_path),
    ]

    print(f"[RUN] power={power} seed={seed} my={args.my_agent} opp={args.opp_agent}")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    result = subprocess.run(cmd)
    return result.returncode


def main():
    args = parse_args()
    root_dir = build_root_dir(args)

    ensure_dirs(root_dir)

    total_missing = 0
    plan = []

    print(
        f"[SCAN] my_agent={args.my_agent} opp_agent={args.opp_agent} "
        f"setup={args.setup} version={normalize_version_tag(args.version)}"
    )

    for power in POWERS:
        power_dir = root_dir / power_dir_name(power)
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

    for power, seed in plan:
        log_path = log_path_for(
            root_dir=root_dir,
            power=power,
            my_agent=args.my_agent,
            opp_agent=args.opp_agent,
            seed=seed,
            version=args.version,
        )
        delete_incomplete_log_if_exists(log_path, dry_run=args.dry_run)

    failed = []
    for power, seed in plan:
        rc = run_one(power, seed, root_dir, args)
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
