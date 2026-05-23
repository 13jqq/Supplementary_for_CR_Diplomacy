#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy

from fairdiplomacy.agents.calculate_multi_results import (
    PROJECT_ROOT,
    normalize_version_tag,
    parse_agents,
    build_root_dir,
    result_csv_path_for,
    read_result_rows_dedup,
    summarize,
    write_outputs,
)


ABLATION_CHOICES = ["full", "no_c12", "no_c34", "no_all"]


def version_with_ablation(version: str, ablation: str) -> str:
    version_tag = normalize_version_tag(version)
    abl = str(ablation or "full").strip().lower()

    if abl == "full":
        return version_tag

    if abl not in ABLATION_CHOICES:
        raise ValueError(f"Unknown ablation={ablation}. Choices={ABLATION_CHOICES}")

    return f"{version_tag}_{abl}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize one multi-random ablation result folder using the same "
            "output format as calculate_multi_results.py."
        )
    )

    parser.add_argument(
        "--agents",
        type=str,
        required=True,
        help="Exactly 4 comma-separated agents, e.g. consistent,cicero_nopress,dipnet,searchbot.",
    )

    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=ABLATION_CHOICES,
        help="Ablation folder suffix: full, no_c12, no_c34, no_all.",
    )

    parser.add_argument(
        "--version",
        default="V1",
        help=(
            "Base version used in the ablation run. "
            "For example, --version V1 --ablation no_c12 reads V1_no_c12."
        ),
    )

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

    parser.add_argument("--seed_start", type=int, default=None)
    parser.add_argument("--seed_end", type=int, default=None)

    parser.add_argument(
        "--bad_end_reasons",
        type=str,
        default="",
        help="Comma-separated end_reason values to exclude from metric summaries.",
    )

    parser.add_argument(
        "--include_bad_end_reasons_in_balance",
        action="store_true",
    )

    parser.add_argument(
        "--write_long_samples",
        action="store_true",
    )

    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    agents = parse_agents(args.agents)

    real_version = version_with_ablation(args.version, args.ablation)

    calc_args = copy.copy(args)
    calc_args.version = real_version

    root_dir = build_root_dir(calc_args, agents)
    result_csv = result_csv_path_for(
        root_dir,
        agents,
        calc_args.version,
        calc_args.csv_name,
    )

    print(f"[ABLATION] {args.ablation}")
    print(f"[BASE_VERSION] {normalize_version_tag(args.version)}")
    print(f"[REAL_VERSION] {real_version}")
    print(f"[ROOT_DIR] {root_dir}")
    print(f"[RESULT_CSV] {result_csv}")

    if not root_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {root_dir}")

    if not result_csv.exists():
        raise FileNotFoundError(f"Result CSV not found: {result_csv}")

    rows = read_result_rows_dedup(result_csv, calc_args)
    if not rows:
        raise RuntimeError(f"No rows found after range filtering in: {result_csv}")

    summary = summarize(rows, agents, calc_args)
    write_outputs(root_dir, result_csv, agents, summary, calc_args)


if __name__ == "__main__":
    main()