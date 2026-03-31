import argparse
import csv
import shutil
from pathlib import Path
from typing import Optional


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
        description="Copy seed subset logs/csvs from *_V1 to *_V2 without modifying source files."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dipnet", action="store_true")
    group.add_argument("--searchbot", action="store_true")
    group.add_argument("--diplodocus_high", action="store_true")
    group.add_argument("--cicero_nopress", action="store_true")

    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=9)
    parser.add_argument(
        "--dst_suffix",
        type=str,
        default="V2",
        help="Destination suffix. Example: log_dipnet_V1 -> log_dipnet_V2",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in destination if they already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned operations, do not actually copy.",
    )
    return parser.parse_args()


def resolve_opp_agent(args) -> str:
    if args.dipnet:
        return "dipnet"
    if args.searchbot:
        return "searchbot"
    if args.diplodocus_high:
        return "diplodocus_high"
    if args.cicero_nopress:
        return "cicero_nopress"
    raise ValueError("No opponent agent selected")


def make_dst_root(src_root: Path, dst_suffix: str) -> Path:
    name = src_root.name
    if name.endswith("_V1"):
        return src_root.parent / f"{name[:-3]}_{dst_suffix}"
    return src_root.parent / f"{name}_{dst_suffix}"


def ensure_dir(path: Path, dry_run: bool = False):
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def log_path_for(root_dir: Path, power: str, opp_agent: str, seed: int) -> Path:
    return root_dir / power / f"run_1v6_my{power}_myconsistent_opp{opp_agent}_seed{seed}.log"


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


def copy_log_if_exists(src: Path, dst: Path, overwrite: bool, dry_run: bool):
    if not src.exists():
        print(f"[MISS LOG] {src}")
        return

    if dst.exists() and not overwrite:
        print(f"[SKIP LOG EXISTS] {dst}")
        return

    print(f"[COPY LOG] {src} -> {dst}")
    if not dry_run:
        ensure_dir(dst.parent, dry_run=False)
        shutil.copy2(src, dst)


def filter_csv_by_seed(src_csv: Path, dst_csv: Path, seed_start: int, seed_end: int, overwrite: bool, dry_run: bool):
    if dst_csv.exists() and not overwrite:
        print(f"[SKIP CSV EXISTS] {dst_csv}")
        return

    with src_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if "seed" not in fieldnames:
            print(f"[SKIP CSV NO SEED] {src_csv}")
            return

        kept_rows = []
        for row in reader:
            raw = row.get("seed", "")
            try:
                seed = int(str(raw).strip())
            except Exception:
                continue
            if seed_start <= seed <= seed_end:
                kept_rows.append(row)

    print(f"[COPY CSV] {src_csv} -> {dst_csv}  kept_rows={len(kept_rows)}")
    if not dry_run:
        ensure_dir(dst_csv.parent, dry_run=False)
        with dst_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)


def maybe_copy_seed_csvs(src_power_dir: Path, dst_power_dir: Path, seed_start: int, seed_end: int, overwrite: bool, dry_run: bool):
    csv_files = sorted(src_power_dir.glob("*.csv"))
    if not csv_files:
        print(f"[NO CSV] {src_power_dir}")
        return

    for src_csv in csv_files:
        if not csv_has_seed_column(src_csv):
            print(f"[SKIP CSV NO SEED] {src_csv}")
            continue
        dst_csv = dst_power_dir / src_csv.name
        filter_csv_by_seed(src_csv, dst_csv, seed_start, seed_end, overwrite, dry_run)


def main():
    args = parse_args()
    opp_agent = resolve_opp_agent(args)

    src_root = ROOT_DIRS[opp_agent]
    dst_root = make_dst_root(src_root, args.dst_suffix)

    print(f"[SRC ROOT] {src_root}")
    print(f"[DST ROOT] {dst_root}")
    print(f"[SEEDS] {args.seed_start}..{args.seed_end}")

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    ensure_dir(dst_root, dry_run=args.dry_run)

    for power in POWERS:
        src_power_dir = src_root / power
        dst_power_dir = dst_root / power

        print(f"\n===== {power} =====")
        if not src_power_dir.exists():
            print(f"[MISS DIR] {src_power_dir}")
            continue

        ensure_dir(dst_power_dir, dry_run=args.dry_run)

        # 1) copy logs for selected seeds
        for seed in range(args.seed_start, args.seed_end + 1):
            src_log = log_path_for(src_root, power, opp_agent, seed)
            dst_log = log_path_for(dst_root, power, opp_agent, seed)
            copy_log_if_exists(src_log, dst_log, args.overwrite, args.dry_run)

        # 2) copy/filter csvs with seed column
        maybe_copy_seed_csvs(
            src_power_dir=src_power_dir,
            dst_power_dir=dst_power_dir,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    print("\n[DONE]")


if __name__ == "__main__":
    main()