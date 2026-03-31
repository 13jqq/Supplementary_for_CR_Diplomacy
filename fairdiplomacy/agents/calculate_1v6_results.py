#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计 1v6 Diplomacy 实验结果（同时统计我方 consistent 与对手 agent）。

目录结构约定：
logs_batch/
  log_dipnet_V1/
    AUSTRIA/
      results_setup=1v6_my=consistent_opp=dipnet_v2.csv
    ENGLAND/
      ...
  log_searchbot_V1/
    ...

脚本会：
1. 扫描 base_dir 下所有对手目录（默认扫描名称以 log_ 开头的目录）
2. 读取其下七个国家文件夹中的结果 CSV
3. 同时统计两类对象：
   - consistent：每局中由我方控制的那个国家（每局 1 个样本）
   - opponent：每局中由对手控制的其余六个国家（每局 6 个样本）
4. 计算以下指标：
   - mean SoS
   - Win / Most SC / Survived / Defeated（四者互斥）
   - C1~C4 触发次数（总次数与平均每个国家样本次数）
   - 支持成功率（sum(success) / sum(total)）
5. 在每个对手目录下输出：
   - summary_metrics_both_agents.csv
   - summary_metrics_both_agents.json
"""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple


POWERS = [
    "AUSTRIA",
    "ENGLAND",
    "FRANCE",
    "GERMANY",
    "ITALY",
    "RUSSIA",
    "TURKEY",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="logs_batch",
        help="包含各对手实验文件夹的大目录，例如 logs_batch",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="log_*",
        help="对手目录匹配模式，默认扫描 log_*",
    )
    return parser.parse_args()


def _to_int(x) -> int:
    if x is None or x == "":
        return 0
    return int(float(x))


def _to_float(x) -> float:
    if x is None or x == "":
        return 0.0
    return float(x)


def _find_country_csvs(opp_dir: Path) -> Dict[str, Path]:
    out = {}
    for power in POWERS:
        power_dir = opp_dir / power
        if not power_dir.is_dir():
            continue

        candidates = sorted(power_dir.glob("results_setup=1v6_my=consistent_opp=*_v2.csv"))
        if not candidates:
            candidates = sorted(power_dir.glob("results_setup=1v6_my=consistent_opp=*.csv"))

        if candidates:
            out[power] = candidates[0]
    return out


def _read_csv_dedup_by_game_id(csv_path: Path) -> List[Dict]:
    """
    读取单个 CSV。
    若同一个 game_id 出现多次，保留最后一行（适合覆盖式重跑后的统计）。
    """
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
    """
    四类结局划分（互斥）：
    1) 若有人 >= 18 SC：
       - 该国 -> Win
       - 其他六国 -> Defeated
    2) 否则（draw）：
       - 最高 SC 的国家 -> Most SC（并列最高也都算）
       - 0 SC -> Defeated
       - 其他且 SC > 0 -> Survived
    """
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
        "n_samples": 0,  # 这里的样本单位是“一个国家在一局中的结果”
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


def summarize_one_opponent(opp_dir: Path) -> Tuple[List[Dict], Dict]:
    country_csvs = _find_country_csvs(opp_dir)

    # role=consistent/opponent, 再分别存 per_power 与 overall
    per_power_stats = {
        "consistent": {p: _init_stats() for p in POWERS},
        "opponent": {p: _init_stats() for p in POWERS},
    }
    overall_stats = {
        "consistent": _init_stats(),
        "opponent": _init_stats(),
    }

    source_files = {}

    for my_power, csv_path in country_csvs.items():
        rows = _read_csv_dedup_by_game_id(csv_path)
        source_files[my_power] = str(csv_path)

        for row in rows:
            row_power = row.get("my_power", "")
            if row_power != my_power:
                continue

            # 1) consistent：该局只有 my_power 这一个国家属于我方
            _update_stats(per_power_stats["consistent"][my_power], row, my_power)
            _update_stats(overall_stats["consistent"], row, my_power)

            # 2) opponent：其余六个国家都属于该对手
            for opp_power in POWERS:
                if opp_power == my_power:
                    continue
                _update_stats(per_power_stats["opponent"][opp_power], row, opp_power)
                _update_stats(overall_stats["opponent"], row, opp_power)

    summary_rows = []

    # 每个角色、每个国家一行
    for role in ["consistent", "opponent"]:
        for power in POWERS:
            finalized = _finalize_stats(per_power_stats[role][power])
            summary_rows.append({
                "role": role,
                "scope": "per_power",
                "power": power,
                **finalized,
            })

    # 每个角色 overall 一行
    for role in ["consistent", "opponent"]:
        finalized = _finalize_stats(overall_stats[role])
        summary_rows.append({
            "role": role,
            "scope": "overall",
            "power": "OVERALL",
            **finalized,
        })

    meta = {
        "opponent_folder": str(opp_dir),
        "country_csvs": source_files,
        "summary_note": (
            "Win/Most SC/Survived/Defeated are mutually exclusive. "
            "If any power has >=18 SC, that power is Win and all others are Defeated. "
            "Otherwise, powers tied for the highest SC are Most SC, powers with 0 SC are Defeated, "
            "and the remaining powers are Survived. "
            "For role=consistent, each game contributes 1 country-sample; "
            "for role=opponent, each game contributes 6 country-samples."
        ),
    }

    return summary_rows, meta


def write_summary_files(opp_dir: Path, summary_rows: List[Dict], meta: Dict):
    csv_path = opp_dir / "summary_metrics_both_agents.csv"
    json_path = opp_dir / "summary_metrics_both_agents.json"

    fieldnames = [
        "role",
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
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "summary_rows": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] saved: {csv_path}")
    print(f"[OK] saved: {json_path}")


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"base_dir not found: {base_dir}")

    opp_dirs = sorted([p for p in base_dir.glob(args.pattern) if p.is_dir()])

    if not opp_dirs:
        print(f"[WARN] no opponent folders found under {base_dir} with pattern={args.pattern}")
        return

    for opp_dir in opp_dirs:
        summary_rows, meta = summarize_one_opponent(opp_dir)
        write_summary_files(opp_dir, summary_rows, meta)


if __name__ == "__main__":
    main()
