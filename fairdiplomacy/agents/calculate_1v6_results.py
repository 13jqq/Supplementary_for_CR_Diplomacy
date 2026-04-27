#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计单个 1v6 / all7 Diplomacy 实验目录下的结果。

适配当前 batch runner 的目录规则：
logs_batch/
  <my_agent>_<setup>/
    log_<my_alias>_vs_<opp_alias>_<Vx>/
      Austria/
        results_setup=1v6_my=cicero_nopress_opp=diplodocus_high_v1.csv
      England/
      ...

功能：
1. 根据 --my_agent / --opp_agent / --setup / --version 精确定位一个实验目录
2. 在每个国家目录中寻找对应结果 CSV（优先匹配与 version 一致的文件名）
3. 同时统计：
   - self: 该局由我方控制的那个国家（每局 1 个 country-sample）
   - opponent: 该局由对手控制的其余国家（1v6 下每局 6 个 country-sample）
4. 输出：
   - summary_metrics_both_agents.csv
   - summary_metrics_both_agents.json
"""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    "diplodocus_high": "diplodocus",
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


def power_dir_name(power: str) -> str:
    return str(power).strip().title()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="logs_batch")
    parser.add_argument("--setup", default="1v6", choices=["1v6", "all7"])
    parser.add_argument("--my_agent", default="consistent", choices=AGENT_CHOICES)
    parser.add_argument("--opp_agent", required=True, choices=AGENT_CHOICES)
    parser.add_argument("--version", default="V1")
    parser.add_argument("--seed_start", type=int, default=None)
    parser.add_argument("--seed_end", type=int, default=None)
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Optional absolute/relative path to a specific experiment dir. If given, it overrides base_dir+my_agent+opp_agent+version resolution.",
    )
    return parser.parse_args()

def _seed_in_range(row: Dict, args) -> bool:
    raw = row.get("seed", "")
    try:
        seed = int(str(raw).strip())
    except Exception:
        return False

    if args.seed_start is not None and seed < args.seed_start:
        return False
    if args.seed_end is not None and seed > args.seed_end:
        return False
    return True

def build_root_dir(args) -> Path:
    if args.experiment_dir:
        p = Path(args.experiment_dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    version_tag = normalize_version_tag(args.version)
    my_dir = f"{args.my_agent}_{args.setup}"
    my_disp = AGENT_FOLDER_ALIAS.get(args.my_agent, args.my_agent)
    opp_disp = AGENT_FOLDER_ALIAS.get(args.opp_agent, args.opp_agent)
    base_dir = resolve_base_dir(args.base_dir)
    return base_dir / my_dir / f"log_{my_disp}_vs_{opp_disp}_{version_tag}"

def _to_int(x) -> int:
    if x is None or x == "":
        return 0
    return int(float(x))


def _to_float(x) -> float:
    if x is None or x == "":
        return 0.0
    return float(x)


def _pick_best_csv(candidates: List[Path], preferred_suffix: str) -> Optional[Path]:
    if not candidates:
        return None

    # 先优先挑选文件名后缀版本匹配的；同类中选最新修改时间
    matched = [p for p in candidates if p.name.endswith(preferred_suffix)]
    if matched:
        return max(matched, key=lambda p: p.stat().st_mtime)

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_country_csvs(root_dir: Path, args) -> Dict[str, Path]:
    out = {}
    preferred_suffix = f"_{version_lower(args.version)}.csv"

    for power in POWERS:
        power_dir = root_dir / power_dir_name(power)
        if not power_dir.is_dir():
            continue

        patterns = [
            f"results_setup={args.setup}_my={args.my_agent}_opp={args.opp_agent}_{version_lower(args.version)}.csv",
            f"results_setup={args.setup}_my={args.my_agent}_opp={args.opp_agent}_*.csv",
            f"results_setup={args.setup}_my={args.my_agent}_opp=*.csv",
            "results_setup=*.csv",
        ]

        candidates: List[Path] = []
        for pat in patterns:
            found = list(power_dir.glob(pat))
            if found:
                candidates = found
                break

        best = _pick_best_csv(candidates, preferred_suffix)
        if best is not None:
            out[power] = best

    return out


def _read_csv_dedup_by_game_id(csv_path: Path) -> List[Dict]:
    """若同一个 game_id 出现多次，保留最后一行。"""
    rows_by_game_id = OrderedDict()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = row.get("game_id")
            if not game_id:
                continue
            rows_by_game_id[game_id] = row
    return list(rows_by_game_id.values())


def _classify_outcome_for_power(row: Dict, power: str) -> str:
    sc_map = {p: _to_int(row.get(f"sc_{p}", 0)) for p in POWERS}
    my_sc = sc_map[power]
    max_sc = max(sc_map.values())

    if max_sc >= 18:
        return "Win" if my_sc >= 18 else "Defeated"

    if my_sc == 0:
        return "Defeated"
    if my_sc == max_sc:
        return "Most SC"
    return "Survived"


def _init_stats() -> Dict:
    return {
        "n_samples": 0,
        "sos_sum": 0.0,
        "win_count": 0,
        "most_sc_count": 0,
        "survived_count": 0,
        "defeated_count": 0,
        "c1_total": 0,
        "c2_total": 0,
        "c3_total": 0,
        "c4_total": 0,
        "support_success_total": 0,
        "support_total_total": 0,
    }


def _update_stats(stats: Dict, row: Dict, power: str):
    stats["n_samples"] += 1
    stats["sos_sum"] += _to_float(row.get(f"sos_{power}", 0.0))

    outcome = _classify_outcome_for_power(row, power)
    if outcome == "Win":
        stats["win_count"] += 1
    elif outcome == "Most SC":
        stats["most_sc_count"] += 1
    elif outcome == "Survived":
        stats["survived_count"] += 1
    elif outcome == "Defeated":
        stats["defeated_count"] += 1

    stats["c1_total"] += _to_int(row.get(f"c1_{power}", 0))
    stats["c2_total"] += _to_int(row.get(f"c2_{power}", 0))
    stats["c3_total"] += _to_int(row.get(f"c3_{power}", 0))
    stats["c4_total"] += _to_int(row.get(f"c4_{power}", 0))

    stats["support_success_total"] += _to_int(row.get(f"support_success_{power}", 0))
    stats["support_total_total"] += _to_int(row.get(f"support_total_{power}", 0))


def _finalize_stats(stats: Dict) -> Dict:
    n = stats["n_samples"]
    if n == 0:
        return {
            "n_samples": 0,
            "mean_sos": 0.0,
            "win_rate": 0.0,
            "most_sc_rate": 0.0,
            "survived_rate": 0.0,
            "defeated_rate": 0.0,
            "c1_total": 0,
            "c2_total": 0,
            "c3_total": 0,
            "c4_total": 0,
            "c1_avg_per_sample": 0.0,
            "c2_avg_per_sample": 0.0,
            "c3_avg_per_sample": 0.0,
            "c4_avg_per_sample": 0.0,
            "support_success_total": 0,
            "support_total_total": 0,
            "support_success_ratio": 0.0,
            "win_count": 0,
            "most_sc_count": 0,
            "survived_count": 0,
            "defeated_count": 0,
        }

    support_ratio = (
        stats["support_success_total"] / stats["support_total_total"]
        if stats["support_total_total"] > 0
        else 0.0
    )

    return {
        "n_samples": n,
        "mean_sos": stats["sos_sum"] / n,
        "win_rate": stats["win_count"] / n,
        "most_sc_rate": stats["most_sc_count"] / n,
        "survived_rate": stats["survived_count"] / n,
        "defeated_rate": stats["defeated_count"] / n,
        "c1_total": stats["c1_total"],
        "c2_total": stats["c2_total"],
        "c3_total": stats["c3_total"],
        "c4_total": stats["c4_total"],
        "c1_avg_per_sample": stats["c1_total"] / n,
        "c2_avg_per_sample": stats["c2_total"] / n,
        "c3_avg_per_sample": stats["c3_total"] / n,
        "c4_avg_per_sample": stats["c4_total"] / n,
        "support_success_total": stats["support_success_total"],
        "support_total_total": stats["support_total_total"],
        "support_success_ratio": support_ratio,
        "win_count": stats["win_count"],
        "most_sc_count": stats["most_sc_count"],
        "survived_count": stats["survived_count"],
        "defeated_count": stats["defeated_count"],
    }


def summarize_one_experiment(root_dir: Path, args) -> Tuple[List[Dict], Dict]:
    country_csvs = _find_country_csvs(root_dir, args)

    per_power_stats = {
        "self": {p: _init_stats() for p in POWERS},
        "opponent": {p: _init_stats() for p in POWERS},
    }
    overall_stats = {
        "self": _init_stats(),
        "opponent": _init_stats(),
    }

    source_files = {}

    for my_power, csv_path in country_csvs.items():
        rows = _read_csv_dedup_by_game_id(csv_path)
        source_files[my_power] = str(csv_path)

        for row in rows:
            if not _seed_in_range(row, args):
                continue

            row_power = str(row.get("my_power", "")).strip().upper()
            if row_power != my_power:
                continue

            _update_stats(per_power_stats["self"][my_power], row, my_power)
            _update_stats(overall_stats["self"], row, my_power)

            for opp_power in POWERS:
                if opp_power == my_power:
                    continue
                _update_stats(per_power_stats["opponent"][opp_power], row, opp_power)
                _update_stats(overall_stats["opponent"], row, opp_power)

    summary_rows = []

    for side, agent_name in [("self", args.my_agent), ("opponent", args.opp_agent)]:
        for power in POWERS:
            finalized = _finalize_stats(per_power_stats[side][power])
            summary_rows.append({
                "side": side,
                "agent_name": agent_name,
                "scope": "per_power",
                "power": power,
                **finalized,
            })

    for side, agent_name in [("self", args.my_agent), ("opponent", args.opp_agent)]:
        finalized = _finalize_stats(overall_stats[side])
        summary_rows.append({
            "side": side,
            "agent_name": agent_name,
            "scope": "overall",
            "power": "OVERALL",
            **finalized,
        })

    meta = {
        "experiment_dir": str(root_dir),
        "setup": args.setup,
        "my_agent": args.my_agent,
        "opp_agent": args.opp_agent,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "version": normalize_version_tag(args.version),
        "country_csvs": source_files,
        "summary_note": (
            "Win/Most SC/Survived/Defeated are mutually exclusive. "
            "If any power has >=18 SC, that power is Win and all others are Defeated. "
            "Otherwise, powers tied for the highest SC are Most SC, powers with 0 SC are Defeated, "
            "and the remaining powers are Survived. "
            "For side=self, each game contributes 1 country-sample; "
            "for side=opponent, each game contributes remaining country-samples."
        ),
    }

    return summary_rows, meta

def build_seed_tag(args) -> str:
    if args.seed_start is None and args.seed_end is None:
        return "allseeds"

    start = "min" if args.seed_start is None else str(args.seed_start)
    end = "max" if args.seed_end is None else str(args.seed_end)
    return f"seed{start}-{end}"

def format_summary_row(row: Dict, ndigits: int = 6) -> Dict:
    float_fields = {
        "mean_sos",
        "win_rate",
        "most_sc_rate",
        "survived_rate",
        "defeated_rate",
        "c1_avg_per_sample",
        "c2_avg_per_sample",
        "c3_avg_per_sample",
        "c4_avg_per_sample",
        "support_success_ratio",
    }

    out = {}
    for k, v in row.items():
        if k in float_fields:
            out[k] = f"{float(v):.{ndigits}f}"
        else:
            out[k] = v
    return out

def write_summary_files(root_dir: Path, summary_rows: List[Dict], meta: Dict, args):
    seed_tag = build_seed_tag(args)
    prefix = f"summary_{args.my_agent}_vs_{args.opp_agent}_{args.setup}_{normalize_version_tag(args.version)}_{seed_tag}"
    csv_path = root_dir / f"{prefix}.csv"
    json_path = root_dir / f"{prefix}.json"

    fieldnames = [
        "side",
        "agent_name",
        "scope",
        "power",
        "n_samples",
        "mean_sos",
        "win_rate",
        "most_sc_rate",
        "survived_rate",
        "defeated_rate",
        "win_count",
        "most_sc_count",
        "survived_count",
        "defeated_count",
        "c1_total",
        "c2_total",
        "c3_total",
        "c4_total",
        "c1_avg_per_sample",
        "c2_avg_per_sample",
        "c3_avg_per_sample",
        "c4_avg_per_sample",
        "support_success_total",
        "support_total_total",
        "support_success_ratio",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(format_summary_row(row, 6))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "summary_rows": [format_summary_row(row, 6) for row in summary_rows],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] saved: {csv_path}")
    print(f"[OK] saved: {json_path}")

def resolve_base_dir(base_dir: str) -> Path:
    p = Path(base_dir)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p

def main():
    args = parse_args()
    root_dir = build_root_dir(args)

    if not root_dir.exists():
        raise FileNotFoundError(f"experiment dir not found: {root_dir}")
    summary_rows, meta = summarize_one_experiment(root_dir, args)
    write_summary_files(root_dir, summary_rows, meta, args)


if __name__ == "__main__":
    main()
