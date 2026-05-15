#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize balanced-random multi-agent Diplomacy experiments.

Expected output layout produced by batch_runner_multi_random.py:

logs_batch/
  multi_random/
    <agents_slug>/
      log_multi_random_<agents_slug>_<Vx>/
        results_setup=multi_agents=<sorted-agent-sig>_v<x>.csv
        assignment_plan_v<x>.jsonl
        block_seed_0000/
          run_multi_block0_shift0_seed0_v1.log
          ...

This script reads the single multi-agent result CSV. Each game row contains:
  - block_seed, shift, seed
  - agent_AUSTRIA ... agent_TURKEY
  - sos_AUSTRIA ... sos_TURKEY
  - sc_AUSTRIA ... sc_TURKEY
  - c1/c2/c3/c4_POWER
  - support_success_POWER / support_total_POWER

It reconstructs per-power country-samples and summarizes performance by agent.
It also outputs assignment balance checks: agent x power counts and agent x power-pair counts.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


# ----------------------------
# Path / argument helpers
# ----------------------------

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


def resolve_project_root(project_root: Optional[str]) -> Path:
    if project_root:
        return Path(project_root).resolve()
    return PROJECT_ROOT


def build_root_dir(args, agents: List[str]) -> Path:
    if args.experiment_dir:
        p = Path(args.experiment_dir)
        return p.resolve() if p.is_absolute() else (resolve_project_root(args.project_root) / p).resolve()

    version_tag = normalize_version_tag(args.version)
    if args.log_root:
        return Path(args.log_root).resolve()

    return (
        resolve_project_root(args.project_root)
        / "logs_batch"
        / "multi_random"
        / agents_slug(agents)
        / f"log_multi_random_{agents_slug(agents)}_{version_tag}"
    )


def result_csv_path_for(root_dir: Path, agents: List[str], version: str, csv_name: Optional[str] = None) -> Path:
    if csv_name:
        p = Path(csv_name)
        return p if p.is_absolute() else root_dir / p
    return root_dir / f"results_setup=multi_agents={agent_sig_for_csv(agents)}_{version_lower(version)}.csv"


def build_seed_tag(args) -> str:
    if args.seed_start is None and args.seed_end is None:
        return "allblocks"
    start = "min" if args.seed_start is None else str(args.seed_start)
    end = "max" if args.seed_end is None else str(args.seed_end)
    return f"block{start}-{end}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize balanced-random multi-agent Diplomacy results."
    )
    parser.add_argument(
        "--agents",
        type=str,
        required=True,
        help="Exactly 4 comma-separated agent kinds, in the same order used by batch_runner_multi_random.py.",
    )
    parser.add_argument("--version", default="V1")
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--log_root", type=str, default=None)
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Optional explicit experiment directory. Overrides default path resolution.",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default=None,
        help="Optional exact result CSV name/path. Relative paths are resolved inside experiment_dir/root_dir.",
    )
    parser.add_argument("--seed_start", type=int, default=None, help="Filter by block_seed lower bound.")
    parser.add_argument("--seed_end", type=int, default=None, help="Filter by block_seed upper bound.")
    parser.add_argument(
        "--bad_end_reasons",
        type=str,
        default="",
        help="Comma-separated end_reason values to exclude from metric summaries, e.g. exception,stopped_unknown.",
    )
    parser.add_argument(
        "--include_bad_end_reasons_in_balance",
        action="store_true",
        help="If set, assignment balance counts include rows whose end_reason is excluded from metrics.",
    )
    parser.add_argument(
        "--write_long_samples",
        action="store_true",
        help="Also write a long-format per-power sample CSV for debugging.",
    )
    return parser.parse_args()


# ----------------------------
# Conversion / math helpers
# ----------------------------

def _to_int(x) -> int:
    if x is None or str(x).strip() == "":
        return 0
    try:
        return int(float(str(x).strip()))
    except Exception:
        return 0


def _to_float(x) -> float:
    if x is None or str(x).strip() == "":
        return 0.0
    try:
        return float(str(x).strip())
    except Exception:
        return 0.0


def _fmt(x, ndigits: int = 6) -> str:
    if isinstance(x, float):
        return f"{x:.{ndigits}f}"
    return str(x)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _sample_sem(xs: List[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var) / math.sqrt(n)


def _ci95(xs: List[float]) -> Tuple[float, float]:
    # Normal approximation. Sufficient for quick experiment reporting.
    if not xs:
        return 0.0, 0.0
    mu = _mean(xs)
    sem = _sample_sem(xs)
    return mu - 1.96 * sem, mu + 1.96 * sem


def _rate(count: int, n: int) -> float:
    return count / n if n else 0.0


# ----------------------------
# Reading / filtering
# ----------------------------

def row_block_seed(row: Dict) -> Optional[int]:
    raw = row.get("block_seed", "")
    if str(raw).strip() != "":
        try:
            return int(float(str(raw).strip()))
        except Exception:
            return None
    return None


def row_shift(row: Dict) -> Optional[int]:
    raw = row.get("shift", "")
    if str(raw).strip() != "":
        try:
            return int(float(str(raw).strip()))
        except Exception:
            return None
    return None


def row_game_seed(row: Dict) -> Optional[int]:
    raw = row.get("seed", "")
    if str(raw).strip() != "":
        try:
            return int(float(str(raw).strip()))
        except Exception:
            return None
    return None


def row_in_block_range(row: Dict, args) -> bool:
    bs = row_block_seed(row)
    # If old rows do not have block_seed, fall back to seed. New multi runner has block_seed.
    if bs is None:
        bs = row_game_seed(row)
    if bs is None:
        return False
    if args.seed_start is not None and bs < args.seed_start:
        return False
    if args.seed_end is not None and bs > args.seed_end:
        return False
    return True


def read_result_rows_dedup(csv_path: Path, args) -> List[Dict]:
    """Read result rows and keep the last row for each (block_seed, shift) or game_id."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_path}")

    rows_by_key: "OrderedDict[Tuple, Dict]" = OrderedDict()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row_in_block_range(row, args):
                continue
            bs = row_block_seed(row)
            sh = row_shift(row)
            if bs is not None and sh is not None:
                key = ("block_shift", bs, sh)
            else:
                key = ("game_id", row.get("game_id", ""))
            if key[-1] == "":
                continue
            rows_by_key[key] = row
    return list(rows_by_key.values())


def parse_bad_end_reasons(s: str) -> set[str]:
    return {x.strip() for x in str(s).split(",") if x.strip()}


# ----------------------------
# Metrics
# ----------------------------

def classify_outcome_for_power(row: Dict, power: str) -> str:
    sc_map = {p: _to_int(row.get(f"sc_{p}", 0)) for p in POWERS}
    my_sc = sc_map[power]
    max_sc = max(sc_map.values()) if sc_map else 0

    if max_sc >= 18:
        return "Win" if my_sc >= 18 else "Defeated"
    if my_sc == 0:
        return "Defeated"
    if my_sc == max_sc:
        return "Most SC"
    return "Survived"


def init_power_stats() -> Dict:
    return {
        "n_samples": 0,
        "sos_values": [],
        "sc_values": [],
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


def update_power_stats(stats: Dict, row: Dict, power: str):
    stats["n_samples"] += 1
    stats["sos_values"].append(_to_float(row.get(f"sos_{power}", 0.0)))
    stats["sc_values"].append(float(_to_int(row.get(f"sc_{power}", 0))))

    outcome = classify_outcome_for_power(row, power)
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


def finalize_power_stats(stats: Dict) -> Dict:
    n = stats["n_samples"]
    sos_values = stats["sos_values"]
    sc_values = stats["sc_values"]
    sos_ci_low, sos_ci_high = _ci95(sos_values)
    sc_ci_low, sc_ci_high = _ci95(sc_values)
    support_ratio = (
        stats["support_success_total"] / stats["support_total_total"]
        if stats["support_total_total"] > 0
        else 0.0
    )
    return {
        "n_samples": n,
        "mean_sos": _mean(sos_values),
        "sem_sos": _sample_sem(sos_values),
        "ci95_sos_low": sos_ci_low,
        "ci95_sos_high": sos_ci_high,
        "mean_sc": _mean(sc_values),
        "sem_sc": _sample_sem(sc_values),
        "ci95_sc_low": sc_ci_low,
        "ci95_sc_high": sc_ci_high,
        "win_rate": _rate(stats["win_count"], n),
        "most_sc_rate": _rate(stats["most_sc_count"], n),
        "survived_rate": _rate(stats["survived_count"], n),
        "defeated_rate": _rate(stats["defeated_count"], n),
        "win_count": stats["win_count"],
        "most_sc_count": stats["most_sc_count"],
        "survived_count": stats["survived_count"],
        "defeated_count": stats["defeated_count"],
        "c1_total": stats["c1_total"],
        "c2_total": stats["c2_total"],
        "c3_total": stats["c3_total"],
        "c4_total": stats["c4_total"],
        "c1_avg_per_sample": stats["c1_total"] / n if n else 0.0,
        "c2_avg_per_sample": stats["c2_total"] / n if n else 0.0,
        "c3_avg_per_sample": stats["c3_total"] / n if n else 0.0,
        "c4_avg_per_sample": stats["c4_total"] / n if n else 0.0,
        "support_success_total": stats["support_success_total"],
        "support_total_total": stats["support_total_total"],
        "support_success_ratio": support_ratio,
    }


def init_agent_game_stats() -> Dict:
    return {
        "n_agent_game_samples": 0,
        "controlled_power_counts": [],
        "sos_sum_values": [],
        "sos_mean_values": [],
        "sc_sum_values": [],
        "sc_mean_values": [],
        "c_total_values": [],
        "support_success_total": 0,
        "support_total_total": 0,
    }


def update_agent_game_stats(stats: Dict, row: Dict, agent: str, powers: List[str]):
    if not powers:
        return
    stats["n_agent_game_samples"] += 1
    stats["controlled_power_counts"].append(float(len(powers)))

    sos_vals = [_to_float(row.get(f"sos_{p}", 0.0)) for p in powers]
    sc_vals = [float(_to_int(row.get(f"sc_{p}", 0))) for p in powers]
    c_total = sum(
        _to_int(row.get(f"c1_{p}", 0))
        + _to_int(row.get(f"c2_{p}", 0))
        + _to_int(row.get(f"c3_{p}", 0))
        + _to_int(row.get(f"c4_{p}", 0))
        for p in powers
    )
    support_success = sum(_to_int(row.get(f"support_success_{p}", 0)) for p in powers)
    support_total = sum(_to_int(row.get(f"support_total_{p}", 0)) for p in powers)

    stats["sos_sum_values"].append(sum(sos_vals))
    stats["sos_mean_values"].append(_mean(sos_vals))
    stats["sc_sum_values"].append(sum(sc_vals))
    stats["sc_mean_values"].append(_mean(sc_vals))
    stats["c_total_values"].append(float(c_total))
    stats["support_success_total"] += support_success
    stats["support_total_total"] += support_total


def finalize_agent_game_stats(stats: Dict) -> Dict:
    n = stats["n_agent_game_samples"]
    support_ratio = (
        stats["support_success_total"] / stats["support_total_total"]
        if stats["support_total_total"] > 0
        else 0.0
    )
    sos_sum_ci = _ci95(stats["sos_sum_values"])
    sos_mean_ci = _ci95(stats["sos_mean_values"])
    return {
        "n_agent_game_samples": n,
        "mean_controlled_powers": _mean(stats["controlled_power_counts"]),
        "mean_sos_sum": _mean(stats["sos_sum_values"]),
        "sem_sos_sum": _sample_sem(stats["sos_sum_values"]),
        "ci95_sos_sum_low": sos_sum_ci[0],
        "ci95_sos_sum_high": sos_sum_ci[1],
        "mean_sos_mean": _mean(stats["sos_mean_values"]),
        "sem_sos_mean": _sample_sem(stats["sos_mean_values"]),
        "ci95_sos_mean_low": sos_mean_ci[0],
        "ci95_sos_mean_high": sos_mean_ci[1],
        "mean_sc_sum": _mean(stats["sc_sum_values"]),
        "mean_sc_mean": _mean(stats["sc_mean_values"]),
        "mean_total_conflicts": _mean(stats["c_total_values"]),
        "support_success_total": stats["support_success_total"],
        "support_total_total": stats["support_total_total"],
        "support_success_ratio": support_ratio,
    }


# ----------------------------
# Main summarization
# ----------------------------

def powers_by_agent(row: Dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for p in POWERS:
        ag = str(row.get(f"agent_{p}", "")).strip()
        if ag:
            out[ag].append(p)
    return dict(out)


def summarize(rows: List[Dict], agents: List[str], args) -> Dict[str, object]:
    bad_end_reasons = parse_bad_end_reasons(args.bad_end_reasons)

    metric_rows = []
    balance_rows_source = []
    long_samples = []

    excluded_rows = []
    for row in rows:
        end_reason = str(row.get("end_reason", "")).strip()
        is_bad = end_reason in bad_end_reasons
        if is_bad:
            excluded_rows.append(row)
        if (not is_bad) or args.include_bad_end_reasons_in_balance:
            balance_rows_source.append(row)
        if not is_bad:
            metric_rows.append(row)

    # Per-power-sample stats: agent overall and agent per power
    overall_by_agent = {a: init_power_stats() for a in agents}
    per_power_by_agent = {(a, p): init_power_stats() for a in agents for p in POWERS}

    # Agent-game aggregation stats
    agent_game_stats = {a: init_agent_game_stats() for a in agents}

    # Assignment balance
    agent_power_counts = Counter()
    agent_pair_counts = Counter()
    agent_singleton_counts = Counter()
    controlled_n_counts = Counter()

    for row in balance_rows_source:
        pba = powers_by_agent(row)
        for ag in agents:
            controlled = sorted(pba.get(ag, []))
            controlled_n_counts[(ag, len(controlled))] += 1
            for p in controlled:
                agent_power_counts[(ag, p)] += 1
            if len(controlled) == 1:
                agent_singleton_counts[(ag, controlled[0])] += 1
            if len(controlled) >= 2:
                for p1, p2 in itertools.combinations(controlled, 2):
                    agent_pair_counts[(ag, p1, p2)] += 1

    for row in metric_rows:
        pba = powers_by_agent(row)
        for p in POWERS:
            ag = str(row.get(f"agent_{p}", "")).strip()
            if not ag:
                continue
            if ag not in overall_by_agent:
                overall_by_agent[ag] = init_power_stats()
            if (ag, p) not in per_power_by_agent:
                per_power_by_agent[(ag, p)] = init_power_stats()
            update_power_stats(overall_by_agent[ag], row, p)
            update_power_stats(per_power_by_agent[(ag, p)], row, p)

            if args.write_long_samples:
                long_samples.append(
                    {
                        "game_id": row.get("game_id", ""),
                        "block_seed": row.get("block_seed", ""),
                        "shift": row.get("shift", ""),
                        "game_seed": row.get("seed", ""),
                        "end_reason": row.get("end_reason", ""),
                        "agent": ag,
                        "power": p,
                        "sos": _to_float(row.get(f"sos_{p}", 0.0)),
                        "sc": _to_int(row.get(f"sc_{p}", 0)),
                        "outcome": classify_outcome_for_power(row, p),
                        "c1": _to_int(row.get(f"c1_{p}", 0)),
                        "c2": _to_int(row.get(f"c2_{p}", 0)),
                        "c3": _to_int(row.get(f"c3_{p}", 0)),
                        "c4": _to_int(row.get(f"c4_{p}", 0)),
                        "support_success": _to_int(row.get(f"support_success_{p}", 0)),
                        "support_total": _to_int(row.get(f"support_total_{p}", 0)),
                    }
                )

        for ag in agents:
            update_agent_game_stats(agent_game_stats[ag], row, ag, sorted(pba.get(ag, [])))

    power_summary_rows = []
    for ag in agents:
        power_summary_rows.append(
            {
                "side": "mixed",
                "agent_name": ag,
                "scope": "overall",
                "power": "OVERALL",
                **finalize_power_stats(overall_by_agent.get(ag, init_power_stats())),
            }
        )
        for p in POWERS:
            power_summary_rows.append(
            {
                "side": "mixed",
                "agent_name": ag,
                "scope": "per_power",
                "power": p,
                **finalize_power_stats(per_power_by_agent.get((ag, p), init_power_stats())),
            }
        )

    agent_game_summary_rows = []
    for ag in agents:
        agent_game_summary_rows.append(
            {
                "agent_name": ag,
                **finalize_agent_game_stats(agent_game_stats.get(ag, init_agent_game_stats())),
            }
        )

    agent_power_balance_rows = []
    expected_per_agent_power = None
    if args.seed_start is not None and args.seed_end is not None:
        expected_per_agent_power = args.seed_end - args.seed_start + 1
    for ag in agents:
        for p in POWERS:
            count = agent_power_counts[(ag, p)]
            agent_power_balance_rows.append(
                {
                    "agent_name": ag,
                    "power": p,
                    "count": count,
                    "expected_count_if_complete": expected_per_agent_power if expected_per_agent_power is not None else "",
                    "delta_from_expected": (count - expected_per_agent_power) if expected_per_agent_power is not None else "",
                }
            )

    pair_balance_rows = []
    for ag in agents:
        for p1, p2 in itertools.combinations(POWERS, 2):
            pair_balance_rows.append(
                {
                    "agent_name": ag,
                    "power_pair": f"{p1}+{p2}",
                    "count": agent_pair_counts[(ag, p1, p2)],
                }
            )

    singleton_balance_rows = []
    for ag in agents:
        for p in POWERS:
            singleton_balance_rows.append(
                {
                    "agent_name": ag,
                    "singleton_power": p,
                    "count": agent_singleton_counts[(ag, p)],
                }
            )

    controlled_n_rows = []
    for ag in agents:
        for n in range(0, len(POWERS) + 1):
            cnt = controlled_n_counts[(ag, n)]
            if cnt:
                controlled_n_rows.append({"agent_name": ag, "n_powers_in_game": n, "count": cnt})

    completed_keys = sorted(
        [
            (row_block_seed(r), row_shift(r), r.get("game_id", ""), r.get("end_reason", ""))
            for r in rows
        ],
        key=lambda x: ((-1 if x[0] is None else x[0]), (-1 if x[1] is None else x[1])),
    )

    return {
        "power_summary_rows": power_summary_rows,
        "agent_game_summary_rows": agent_game_summary_rows,
        "agent_power_balance_rows": agent_power_balance_rows,
        "pair_balance_rows": pair_balance_rows,
        "singleton_balance_rows": singleton_balance_rows,
        "controlled_n_rows": controlled_n_rows,
        "long_samples": long_samples,
        "meta": {
            "n_rows_read_after_dedup_and_range_filter": len(rows),
            "n_metric_rows": len(metric_rows),
            "n_balance_rows_source": len(balance_rows_source),
            "n_excluded_bad_end_reason_rows": len(excluded_rows),
            "bad_end_reasons_excluded_from_metrics": sorted(list(bad_end_reasons)),
            "completed_keys": completed_keys,
            "metric_note": (
                "power_summary_rows use country-samples: every controlled power is one sample. "
                "agent_game_summary_rows aggregate all powers controlled by the same agent in the same game. "
                "mean_sos_sum is useful if treating all powers controlled by an agent in a game as a combined allocation; "
                "mean_sos_mean is useful if normalizing by number of controlled powers in that game."
            ),
            "outcome_note": (
                "Win/Most SC/Survived/Defeated are mutually exclusive at the power level. "
                "If any power has >=18 SC, that power is Win and all others are Defeated. "
                "Otherwise, powers tied for highest SC are Most SC, powers with 0 SC are Defeated, "
                "and the remaining powers are Survived."
            ),
        },
    }


# ----------------------------
# Writing
# ----------------------------

def write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: (_fmt(v) if isinstance(v, float) else v) for k, v in row.items()}
            writer.writerow(out)


def write_outputs(root_dir: Path, result_csv: Path, agents: List[str], summary: Dict[str, object], args):
    seed_tag = build_seed_tag(args)
    prefix = f"summary_multi_random_{agents_slug(agents)}_{normalize_version_tag(args.version)}_{seed_tag}"

    power_summary_path = root_dir / f"{prefix}_power_samples.csv"
    agent_game_summary_path = root_dir / f"{prefix}_agent_game_samples.csv"
    balance_power_path = root_dir / f"{prefix}_balance_agent_power.csv"
    balance_pair_path = root_dir / f"{prefix}_balance_agent_pair.csv"
    balance_singleton_path = root_dir / f"{prefix}_balance_singleton.csv"
    balance_controlled_n_path = root_dir / f"{prefix}_balance_controlled_n.csv"
    meta_path = root_dir / f"{prefix}.json"

    power_fieldnames = [
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
    agent_game_fieldnames = [
        "agent_name",
        "n_agent_game_samples",
        "mean_controlled_powers",
        "mean_sos_sum",
        "sem_sos_sum",
        "ci95_sos_sum_low",
        "ci95_sos_sum_high",
        "mean_sos_mean",
        "sem_sos_mean",
        "ci95_sos_mean_low",
        "ci95_sos_mean_high",
        "mean_sc_sum",
        "mean_sc_mean",
        "mean_total_conflicts",
        "support_success_total",
        "support_total_total",
        "support_success_ratio",
    ]

    write_csv(power_summary_path, summary["power_summary_rows"], power_fieldnames)
    write_csv(agent_game_summary_path, summary["agent_game_summary_rows"], agent_game_fieldnames)
    write_csv(balance_power_path, summary["agent_power_balance_rows"])
    write_csv(balance_pair_path, summary["pair_balance_rows"])
    write_csv(balance_singleton_path, summary["singleton_balance_rows"])
    write_csv(balance_controlled_n_path, summary["controlled_n_rows"])

    if args.write_long_samples:
        long_path = root_dir / f"{prefix}_long_power_samples.csv"
        write_csv(long_path, summary["long_samples"])
    else:
        long_path = None

    meta = {
        "experiment_dir": str(root_dir),
        "result_csv": str(result_csv),
        "agents": agents,
        "version": normalize_version_tag(args.version),
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        **summary["meta"],
        "output_files": {
            "power_samples_summary": str(power_summary_path),
            "agent_game_samples_summary": str(agent_game_summary_path),
            "balance_agent_power": str(balance_power_path),
            "balance_agent_pair": str(balance_pair_path),
            "balance_singleton": str(balance_singleton_path),
            "balance_controlled_n": str(balance_controlled_n_path),
            "long_power_samples": str(long_path) if long_path else None,
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {power_summary_path}")
    print(f"[OK] saved: {agent_game_summary_path}")
    print(f"[OK] saved: {balance_power_path}")
    print(f"[OK] saved: {balance_pair_path}")
    print(f"[OK] saved: {balance_singleton_path}")
    print(f"[OK] saved: {balance_controlled_n_path}")
    if long_path:
        print(f"[OK] saved: {long_path}")
    print(f"[OK] saved: {meta_path}")

    # Console leaderboard: country-sample overall rows only.
    print("\n[LEADERBOARD: country-sample overall mean_sos]")
    overall = [r for r in summary["power_summary_rows"] if r.get("scope") == "overall"]
    overall = sorted(overall, key=lambda r: float(r.get("mean_sos", 0.0)), reverse=True)
    for r in overall:
        print(
            f"  {r['agent_name']:<28} "
            f"mean_sos={r['mean_sos']:.6f} ± {r['sem_sos']:.6f} "
            f"n={r['n_samples']} mean_sc={r['mean_sc']:.3f} support={r['support_success_ratio']:.4f}"
        )

    print("\n[BALANCE CHECK: agent x power counts]")
    # Print compact table.
    counts = defaultdict(dict)
    for r in summary["agent_power_balance_rows"]:
        counts[r["agent_name"]][r["power"]] = r["count"]
    header = "agent".ljust(28) + " ".join(p[:3].rjust(4) for p in POWERS)
    print(header)
    for ag in agents:
        vals = " ".join(str(counts[ag].get(p, 0)).rjust(4) for p in POWERS)
        print(ag.ljust(28) + vals)


def main():
    args = parse_args()
    agents = parse_agents(args.agents)
    root_dir = build_root_dir(args, agents)
    result_csv = result_csv_path_for(root_dir, agents, args.version, args.csv_name)

    if not root_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {root_dir}")
    if not result_csv.exists():
        raise FileNotFoundError(f"Result CSV not found: {result_csv}")

    rows = read_result_rows_dedup(result_csv, args)
    if not rows:
        raise RuntimeError(f"No rows found after range filtering in: {result_csv}")

    summary = summarize(rows, agents, args)
    write_outputs(root_dir, result_csv, agents, summary, args)


if __name__ == "__main__":
    main()
