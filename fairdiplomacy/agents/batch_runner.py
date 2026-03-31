import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


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
    parser = argparse.ArgumentParser()
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



def try_claim_log(log_path: Path, power: str, seed: int, opp_agent: str) -> bool:
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "power": power,
        "seed": seed,
        "opp_agent": opp_agent,
        "log_path": str(log_path),
    }

    try:
        fd = os.open(str(log_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w") as f:
        f.write("[CLAIMED BY BATCH_RUNNER]\n")
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return True




def find_and_claim_next_task(root_dir: Path, opp_agent: str, seed_start: int, seed_end: int):
    for power in POWERS:
        for seed in range(seed_start, seed_end + 1):
            log_path = log_path_for(root_dir, power, opp_agent, seed)

            # 已有 log：直接跳过
            if log_path.exists():
                continue

            # 尝试原子创建 log 作为占位
            ok = try_claim_log(log_path, power, seed, opp_agent)
            if ok:
                return {
                    "power": power,
                    "seed": seed,
                    "log_path": log_path,
                }

    return None

def run_one_task(task, args, opp_agent: str, root_dir: Path) -> int:
    power = task["power"]
    seed = task["seed"]
    log_path = task["log_path"]

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
    result = subprocess.run(cmd)
    return result.returncode


def main():
    args = parse_args()
    opp_agent = resolve_opp_agent(args)
    root_dir = ROOT_DIRS[opp_agent]

    ensure_dirs(root_dir)

    while True:
        task = find_and_claim_next_task(
            root_dir=root_dir,
            opp_agent=opp_agent,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
        )

        if task is None:
            print(f"[DONE] no remaining tasks for opp_agent={opp_agent}")
            break

        power = task["power"]
        seed = task["seed"]
        log_path = task["log_path"]

        try:
            rc = run_one_task(task, args, opp_agent, root_dir)
            if rc != 0:
                with open(log_path, "a") as f:
                    f.write(f"\n[BATCH_RUNNER_ERROR] returncode={rc}\n")
                print(f"[FAILED] power={power} seed={seed} returncode={rc}")
                sys.exit(rc)

            print(f"[OK] power={power} seed={seed}")

        except KeyboardInterrupt:
            print(f"[STOPPED] power={power} seed={seed}")
            raise
        except Exception as e:
            with open(log_path, "a") as f:
                f.write(f"\n[BATCH_RUNNER_EXCEPTION] {repr(e)}\n")
            print(f"[EXCEPTION] power={power} seed={seed} err={e}")
            raise


if __name__ == "__main__":
    main()