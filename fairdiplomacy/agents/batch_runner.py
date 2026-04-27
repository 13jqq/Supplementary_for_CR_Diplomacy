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


# 仅用于文件夹显示名；csv/log 里仍保留真实 agent 名称
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
    parser = argparse.ArgumentParser()

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


def try_claim_log(log_path: Path, power: str, seed: int, my_agent: str, opp_agent: str, version: str) -> bool:
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "power": power,
        "seed": seed,
        "my_agent": my_agent,
        "opp_agent": opp_agent,
        "version": normalize_version_tag(version),
        "log_path": str(log_path),
    }

    try:
        fd = os.open(str(log_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("[CLAIMED BY BATCH_RUNNER]\n")
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return True


def find_and_claim_next_task(root_dir: Path, my_agent: str, opp_agent: str, version: str, seed_start: int, seed_end: int):
    for power in POWERS:
        for seed in range(seed_start, seed_end + 1):
            log_path = log_path_for(root_dir, power, my_agent, opp_agent, seed, version)

            if log_path.exists():
                continue

            ok = try_claim_log(log_path, power, seed, my_agent, opp_agent, version)
            if ok:
                return {
                    "power": power,
                    "seed": seed,
                    "log_path": log_path,
                }

    return None


def run_one_task(task, args, root_dir: Path) -> int:
    power = task["power"]
    seed = task["seed"]
    log_path = task["log_path"]
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
    result = subprocess.run(cmd)
    return result.returncode


def main():
    args = parse_args()
    root_dir = build_root_dir(args)
    print(f"[ROOT_DIR] {root_dir}")

    ensure_dirs(root_dir)

    while True:
        task = find_and_claim_next_task(
            root_dir=root_dir,
            my_agent=args.my_agent,
            opp_agent=args.opp_agent,
            version=args.version,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
        )

        if task is None:
            print(f"[DONE] no remaining tasks for my_agent={args.my_agent}, opp_agent={args.opp_agent}")
            break

        power = task["power"]
        seed = task["seed"]
        log_path = task["log_path"]

        try:
            rc = run_one_task(task, args, root_dir)
            if rc != 0:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n[BATCH_RUNNER_ERROR] returncode={rc}\n")
                print(f"[FAILED] power={power} seed={seed} returncode={rc}")
                sys.exit(rc)

            print(f"[OK] power={power} seed={seed}")

        except KeyboardInterrupt:
            print(f"[STOPPED] power={power} seed={seed}")
            raise
        except Exception as e:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[BATCH_RUNNER_EXCEPTION] {repr(e)}\n")
            print(f"[EXCEPTION] power={power} seed={seed} err={e}")
            raise


if __name__ == "__main__":
    main()