# coding=utf-8
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import heyhi
import os
import csv
try:
    from .consistent_agent import POWERS, get_territory_parts, ConsistentAgent
except Exception:
    from fairdiplomacy.agents.consistent_agent import POWERS, get_territory_parts, ConsistentAgent
import heyhi
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent
from fairdiplomacy.agents.searchbot_agent import SearchBotAgent
from fairdiplomacy.agents.base_strategy_model_agent import BaseStrategyModelAgent
import time

STOP_AFTER_PHASE = "W1913A"   # ✅ 处理完 W1913A 立刻停止
N_SCS = 34                    # 给 compute_game_sos_from_state 用

# ----------------------------
# Utils
# ----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def compute_game_sos_from_state(game_state: Dict) -> List[float]:
    center_counts = [len(game_state["centers"].get(p, [])) for p in POWERS]
    clear_wins = [c > (N_SCS / 2) for c in center_counts]
    if any(clear_wins):
        return [float(w) for w in clear_wins]
    center_squares = [x ** 2 for x in center_counts]
    sum_sq = sum(center_squares)
    return [c / sum_sq for c in center_squares]
def is_game_done(game: Any) -> bool:
    for attr in ("is_game_done", "is_game_over", "game_over"):
        if hasattr(game, attr):
            try:
                v = getattr(game, attr)
                return bool(v() if callable(v) else v)
            except Exception:
                pass
    ph = str(game.get_current_phase()).upper()
    return ("COMPLETED" in ph) or (ph in {"DONE", "END"})


def _disable_cfr_messages_best_effort(agent_cfg: Any) -> Any:
    """
    保险：某些 cfg 会残留 cfr_messages，SearchBotAgent 会 assert not self.cfr_messages。
    这里做 best-effort 清理（不依赖具体 proto 结构，存在就清掉）。
    """
    try:
        cfg = agent_cfg.to_editable()
    except Exception:
        return agent_cfg

    # 常见结构：cfg.base_searchbot_cfg.cfr_messages
    try:
        bscfg = cfg.base_searchbot_cfg
        # bool / message / repeated 都试一下
        for fld in ("cfr_messages", "cfr_message", "cfr_message_cfg", "cfr_message_generation"):
            try:
                bscfg.ClearField(fld)
            except Exception:
                pass
        try:
            setattr(bscfg, "cfr_messages", False)
        except Exception:
            pass
    except Exception:
        pass

    try:
        return cfg.to_frozen()
    except Exception:
        return agent_cfg

def load_consistent_agent(cfg_path: str, *, skip_cache: bool = False) -> ConsistentAgent:
    full_cfg = heyhi.load_config(cfg_path)

    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "consistent_agent"):
        agent_cfg = full_cfg.agent.consistent_agent
    elif hasattr(full_cfg, "consistent_agent"):
        agent_cfg = full_cfg.consistent_agent
    else:
        raise ValueError(f"Bad config structure in {cfg_path}: cannot find consistent_agent")

    return ConsistentAgent(agent_cfg, skip_base_strategy_model_cache=skip_cache)

def load_cicero_nopress_agent(cfg_path: str, *, skip_cache: bool = False) -> BQRE1PAgent:
    """
    Cicero_nopress: conf/common/agents/cicero_nopress.prototxt
    该配置应包含 bqre1p {...}（仅决策部分），直接用 BQRE1PAgent 加载。
    """
    full_cfg = heyhi.load_config(cfg_path)

    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "bqre1p"):
        agent_cfg = full_cfg.agent.bqre1p
    elif hasattr(full_cfg, "bqre1p"):
        agent_cfg = full_cfg.bqre1p
    else:
        raise ValueError(f"Bad config structure in {cfg_path}: cannot find bqre1p")

    return BQRE1PAgent(agent_cfg, skip_base_strategy_model_cache=skip_cache)


def load_diplodocus_high_agent(cfg_path: str, *, skip_cache: bool = False) -> BQRE1PAgent:
    """
    Diplodocus-High: conf/common/agents/diplodocus_high.prototxt

    你给的配置结构非常明确：includes mount 到 bqre1p.base_searchbot_cfg，
    且文件内存在 bqre1p { base_searchbot_cfg { ... } ... }。
    因此它本质就是 “一个 bqre1p 配置”，必须用 BQRE1PAgent 加载。
    """
    full_cfg = heyhi.load_config(cfg_path)

    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "bqre1p"):
        agent_cfg = full_cfg.agent.bqre1p
    elif hasattr(full_cfg, "bqre1p"):
        agent_cfg = full_cfg.bqre1p
    else:
        raise ValueError(f"Bad config structure in {cfg_path}: cannot find bqre1p")

    return BQRE1PAgent(agent_cfg, skip_base_strategy_model_cache=skip_cache)

def load_diplodocus_low_agent(cfg_path: str, *, skip_cache: bool = False) -> BQRE1PAgent:
    """
    Diplodocus-Low: conf/common/agents/diplodocus_low.prototxt
    与 diplodocus_high 完全一致，只改名字（同为 bqre1p 配置，用 BQRE1PAgent 加载）。
    """
    full_cfg = heyhi.load_config(cfg_path)

    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "bqre1p"):
        agent_cfg = full_cfg.agent.bqre1p
    elif hasattr(full_cfg, "bqre1p"):
        agent_cfg = full_cfg.bqre1p
    else:
        raise ValueError(f"Bad config structure in {cfg_path}: cannot find bqre1p")

    return BQRE1PAgent(agent_cfg, skip_base_strategy_model_cache=skip_cache)

def load_searchbot_agent(cfg_path: str, *, skip_cache: bool = False) -> SearchBotAgent:
    """
    SearchBot: conf/common/agents/searchbot.prototxt
    该配置应包含 searchbot {...}，用 SearchBotAgent 加载。
    """
    full_cfg = heyhi.load_config(cfg_path)

    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "searchbot"):
        agent_cfg = full_cfg.agent.searchbot
    elif hasattr(full_cfg, "searchbot"):
        agent_cfg = full_cfg.searchbot
    else:
        raise ValueError(f"Bad config structure in {cfg_path}: cannot find searchbot")

    return SearchBotAgent(agent_cfg, skip_base_strategy_model_cache=skip_cache)


def load_dipnet_agent(cfg_path: str) -> BaseStrategyModelAgent:
    """
    DipNet（你这里定义为 blueprint/base strategy model）:
    conf/common/agents/base_strategy_model.prototxt
    文件里应包含 base_strategy_model { model_path: "models/blueprint.pt" ... }
    按你之前能跑通的方式：BaseStrategyModelAgent(cfg) —— 不加 try。
    """
    full_cfg = heyhi.load_config(cfg_path)

    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "base_strategy_model"):
        cfg = full_cfg.agent.base_strategy_model
    elif hasattr(full_cfg, "base_strategy_model"):
        cfg = full_cfg.base_strategy_model
    else:
        raise ValueError(f"Bad config structure in {cfg_path}: cannot find base_strategy_model")

    return BaseStrategyModelAgent(cfg)

# ----------------------------
# Agent assignment
# ----------------------------
AgentKind = str
AGENT_KINDS: List[AgentKind] = ["consistent", "cicero_nopress", "diplodocus_high","diplodocus_low", "searchbot", "dipnet"]


def pick_agent_for_power(
    setup: str,
    my_power: str,
    my_agent_kind: AgentKind,
    opp_agent_kind: AgentKind,
    all_agent_kind: AgentKind,
    agent_pool: Dict[AgentKind, Any],
) -> Dict[str, Any]:
    """
    setup:
      - "1v6": my_power 用 my_agent_kind，其它用 opp_agent_kind
      - "all7": 七国都用 all_agent_kind
    """
    m: Dict[str, Any] = {}
    for p in POWERS:
        if setup == "all7":
            m[p] = agent_pool[all_agent_kind]
        elif setup == "1v6":
            m[p] = agent_pool[my_agent_kind] if p == my_power else agent_pool[opp_agent_kind]
        else:
            raise ValueError(f"Unknown setup: {setup}")
    return m


def choose_orders_wrapper(
    agent: Any,
    game: Any,
    power: str,
    state: Any,
    *,
    source: str,
    top_k: int,
    mode: str,
) -> Tuple[
    List[str],
    List[Tuple[Any, float]],
    str,
    List[Tuple[Any, float, str]],
    List[Tuple[Any, float]],
]:
    if hasattr(agent, "get_orders_info"):
        info = agent.get_orders_info(
            game=game,
            power=power,
            state=state,
            source=source,
            top_k=top_k,
            mode=mode,
        )
        raw_items = info.get("raw_items", []) or []
        return info["orders"], info["items"], info["used_source"], info["dropped"], raw_items

    orders = agent.get_orders(game, power=power, state=state)
    return orders, [], type(agent).__name__, [], []

def _fmt_action(action: Any) -> str:
    """把候选动作转成便于日志查看的字符串"""
    if isinstance(action, (list, tuple)):
        return "[" + ", ".join(map(str, action)) + "]"
    return str(action)


def _is_empty_action(action: Any) -> bool:
    """判断一个候选动作是不是空动作"""
    if action is None:
        return True
    if isinstance(action, (list, tuple)):
        return len(action) == 0
    return str(action).strip() == ""


def _is_empty_orders(orders: List[str]) -> bool:
    """判断最终 orders 是不是空"""
    return (not orders) or all(str(x).strip() == "" for x in orders)


def _dump_candidates(f, power, raw_items, items, dropped, orders):
    """把这次的原始候选、保留候选、过滤候选、最终选择全部写进日志"""
    f.write(f"[RAW CANDIDATES] power={power} n={len(raw_items)}\n")
    for j, (a, p) in enumerate(raw_items):
        f.write(f"  R{j:02d}  p={float(p):.8f}  action={_fmt_action(a)}\n")

    f.write(f"[KEPT CANDIDATES] power={power} n={len(items)}\n")
    for j, (a, p) in enumerate(items):
        f.write(f"  K{j:02d}  p={float(p):.8f}  action={_fmt_action(a)}\n")

    f.write(f"[FILTERED OUT] power={power} n={len(dropped)}\n")
    for j, (a, p, reason) in enumerate(dropped):
        f.write(
            f"  D{j:02d}  p={float(p):.8f}  reason={reason}  action={_fmt_action(a)}\n"
        )

    f.write(f"[FINAL CHOICE] power={power} orders={orders}\n")

def _check_consistent_candidate_bug_or_raise(
    f,
    phase_str: str,
    power: str,
    raw_items,
    items,
    dropped,
    orders,
):
    """
    检查 consistent agent 的候选动作是否异常。

    注意：
    - A 阶段（Winter Adjustment）允许空动作
    - R 阶段（Retreat）允许空动作
    这两个阶段直接跳过验证
    - 如果 consistency filter 把 raw_items 全部过滤掉，但代码已经 fallback 到 raw_items，
      这不算异常，只记日志
    """
    # A=Winter Adjustment, R=Retreat
    if phase_str.endswith("A") or phase_str.endswith("R"):
        return

    reasons = []

    raw_empty = (len(raw_items) == 0)
    items_empty = (len(items) == 0)
    raw_all_empty_action = (len(raw_items) > 0 and all(_is_empty_action(a) for a, _ in raw_items))
    items_all_empty_action = (len(items) > 0 and all(_is_empty_action(a) for a, _ in items))
    all_raw_filtered = (len(raw_items) > 0 and len(dropped) == len(raw_items))

    # 1. 原始候选集为空：非常不应该
    if raw_empty:
        reasons.append("RAW_ITEMS_EMPTY")

    # 2. 原始候选集非空，但全是空动作
    if raw_all_empty_action:
        reasons.append("RAW_ITEMS_ALL_EMPTY_ACTION")

    # 3. 所有 raw_items 都被 filter 掉了
    #    如果 items == raw_items，说明已经按设计 fallback，不算异常，只记日志
    if all_raw_filtered:
        if items == raw_items and not items_empty:
            f.write(
                f"[FILTER FALLBACK] phase={phase_str} power={power} "
                f"all raw_items were filtered, fallback to raw_items\n"
            )
        else:
            reasons.append("ALL_RAW_ITEMS_FILTERED_NO_FALLBACK")

    # 4. 最终 items 为空：这才是真异常
    if items_empty:
        reasons.append("ITEMS_EMPTY_AFTER_FILTER")

    # 5. 最终保留下来的 items 全是空动作
    if items_all_empty_action:
        reasons.append("ITEMS_ALL_EMPTY_ACTION")

    # 6. 明明最终 items 里有非空动作，但最后 orders 还是空
    #    这种更值得报错
    if len(items) > 0 and (not items_all_empty_action) and _is_empty_orders(orders):
        reasons.append("FINAL_ORDERS_EMPTY_WITH_NONEMPTY_ITEMS")

    if reasons:
        f.write("\n" + "!" * 90 + "\n")
        f.write(
            f"[CANDIDATE BUG] phase={phase_str} power={power} reasons={','.join(reasons)}\n"
        )
        _dump_candidates(f, power, raw_items, items, dropped, orders)
        f.write("!" * 90 + "\n")
        f.flush()
        raise RuntimeError(
            f"[{phase_str}][{power}] candidate bug: {','.join(reasons)}"
        )
def _fmt_post_check_violations(violations):
    order = ["C1", "C2", "C3", "C4"]
    grouped = {k: [] for k in order}

    for tag, detail in violations:
        if detail is None:
            continue

        if isinstance(detail, (list, tuple, set)):
            vals = [str(x).strip() for x in detail if str(x).strip()]
        else:
            s = str(detail).strip()
            vals = [s] if s else []

        grouped.setdefault(tag, []).extend(vals)

    parts = []
    seen_tags = {tag for tag, _ in violations}

    for tag in order:
        if tag not in seen_tags:
            continue

        dedup = []
        seen = set()
        for x in grouped.get(tag, []):
            if x not in seen:
                seen.add(x)
                dedup.append(x)

        if dedup:
            parts.append(f"{tag}[{' ; '.join(dedup)}]")
        else:
            parts.append(tag)

    return ", ".join(parts)
#  ----------------------------
# Main
# ----------------------------
def main():
    import argparse
    import os
    from datetime import datetime
    from fairdiplomacy import pydipcc

    parser = argparse.ArgumentParser()

    # ✅ consistent agent config（你自己的）
    parser.add_argument("--cfg_consistent", type=str, default="conf/common/agents/consistent_agent.prototxt")

    # ✅ 4 个对手 config（按你给的“正确写法”命名）
    parser.add_argument("--cfg_cicero_nopress", type=str, default="conf/common/agents/cicero_nopress.prototxt")
    parser.add_argument("--cfg_diplodocus_high", type=str, default="conf/common/agents/diplodocus_high.prototxt")
    parser.add_argument("--cfg_diplodocus_low", type=str, default="conf/common/agents/diplodocus_low.prototxt")
    parser.add_argument("--cfg_searchbot", type=str, default="conf/common/agents/searchbot.prototxt")
    parser.add_argument("--cfg_dipnet", type=str, default="conf/common/agents/base_strategy_model.prototxt")

    parser.add_argument("--project_root", type=str, default="/workspace/Diplomacy/diplomacy_cicero")
    parser.add_argument("--power", type=str, default="AUSTRIA", choices=POWERS)
    parser.add_argument("--setup", type=str, default="1v6", choices=["1v6", "all7"])
    parser.add_argument("--seed", type=int, default=0)

    # ✅ 策略选择：包含我们自己 + 4 个对手
    # 默认先测 DipNet：opp_agent=dipnet
    AGENT_TYPES = ["consistent", "cicero_nopress", "diplodocus_high","diplodocus_low", "searchbot", "dipnet"]
    parser.add_argument("--my_agent", type=str, default="consistent", choices=AGENT_TYPES)
    parser.add_argument("--opp_agent", type=str, default="dipnet", choices=AGENT_TYPES)
    parser.add_argument("--all_agent", type=str, default="consistent", choices=AGENT_TYPES)
        # only affects consistent branch
    parser.add_argument("--source", type=str, default="bqre_topK", choices=["bqre_topK", "search_br", "bp"])
    parser.add_argument("--mode", type=str, default="bqre", choices=["top1", "sample", "bqre"])
    parser.add_argument("--topk", type=int, default=30)

    parser.add_argument("--max_phases", type=int, default=60)
    parser.add_argument("--log_dir", type=str, default="logs_consistent")
    parser.add_argument("--log", type=str, default=None)

    args = parser.parse_args()
    seed_everything(args.seed)

    # cd to project root for relative conf/model paths
    if args.project_root and os.path.exists(args.project_root):
        os.chdir(args.project_root)

    # log path
    ts = datetime.now().strftime("%y%m%d%H%M%S")
    log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(os.getcwd(), args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    default_name = (
        f"run_{args.setup}_my{args.power}_my{args.my_agent}_"
        f"opp{args.opp_agent}_all{args.all_agent}_seed{args.seed}.log"
    )
    log_path = args.log if args.log else os.path.join(log_dir, default_name)

    # ---- decide which agent kinds are actually needed in THIS run ----
    needed: set[str] = set()
    if args.setup == "all7":
        needed.add(args.all_agent)
    else:
        needed.add(args.my_agent)
        needed.add(args.opp_agent)
    needed.add("consistent")

    # BOOT log early
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== BOOT ===\n")
        f.write(f"needed={sorted(list(needed))}\n")
        f.write("Loading agents...\n")
        f.flush()

    # lazy-load agent pool
    agent_pool: Dict[str, Any] = {}

    def _load_one(t: str):
        if t == "consistent":
            return load_consistent_agent(args.cfg_consistent, skip_cache=False)

        if t == "cicero_nopress":
            return load_cicero_nopress_agent(args.cfg_cicero_nopress, skip_cache=False)

        if t == "diplodocus_high":
            return load_diplodocus_high_agent(args.cfg_diplodocus_high, skip_cache=False)

        if t == "diplodocus_low":
            return load_diplodocus_low_agent(args.cfg_diplodocus_low, skip_cache=False)


        if t == "searchbot":
            return load_searchbot_agent(args.cfg_searchbot, skip_cache=False)

        if t == "dipnet":
            return load_dipnet_agent(args.cfg_dipnet)

        raise ValueError(f"Unknown agent type: {t}")

    for k in sorted(list(needed)):
        agent_pool[k] = _load_one(k)

    game = pydipcc.Game()

    # assign agents
    agent_for_power = pick_agent_for_power(
        setup=args.setup,
        my_power=args.power,
        my_agent_kind=args.my_agent,
        opp_agent_kind=args.opp_agent,
        all_agent_kind=args.all_agent,
        agent_pool=agent_pool,
    )
    states: Dict[str, Any] = {p: agent_for_power[p].initialize_state(p) for p in POWERS}
    def _tag(pwr: str) -> str:
        ag = agent_for_power[pwr]
        for k, v in agent_pool.items():
            if ag is v:
                return k
        return type(ag).__name__

   

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== RUN START ===\n")
        f.write(f"cwd={os.getcwd()}\n")
        f.write(f"setup={args.setup} my_power={args.power}\n")
        f.write(f"cfg_consistent={args.cfg_consistent}\n")
        f.write(f"cfg_cicero_nopress={args.cfg_cicero_nopress}\n")
        f.write(f"cfg_diplodocus_high={args.cfg_diplodocus_high}\n")
        f.write(f"cfg_diplodocus_low={args.cfg_diplodocus_low}\n")
        f.write(f"cfg_searchbot={args.cfg_searchbot}\n")
        f.write(f"cfg_dipnet={args.cfg_dipnet}\n")
        f.write(f"my_agent={args.my_agent} opp_agent={args.opp_agent} all_agent={args.all_agent}\n")
        f.write(f"seed={args.seed} source={args.source} mode={args.mode} topk={args.topk} max_phases={args.max_phases}\n")

        f.write("[ASSIGNMENT]\n")
        for p in POWERS:
            f.write(f"  {p}: {_tag(p)}\n")
        f.write("\n")
        f.flush()
        end_reason = None
        last_processed_phase = None   # 记录最后一个“被 process 的 phase”，用于 summary 展示
        run_error = None              # (phase_str, repr(e))

        support_total = {p: 0 for p in POWERS}
        support_success = {p: 0 for p in POWERS}

        # ✅ 同一组实验写到同一张表
        csv_path = os.path.join(
            log_dir,
            f"results_setup={args.setup}_my={args.my_agent}_opp={args.opp_agent}_v2.csv",
        )
        game_id = os.path.splitext(os.path.basename(log_path))[0]

        step = 0
        total_check_counts = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "NONE": 0}
        per_power_check_counts = {
            p: {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "NONE": 0} for p in POWERS
        }

        while step < args.max_phases and not is_game_done(game):
            phase_t0 = time.perf_counter()
            phase = game.get_current_phase()
            phase_str = str(phase).upper()
            last_processed_phase = phase_str

            # state snapshot for logging
            try:
                st = game.get_state()
            except Exception:
                st = {}
            if not isinstance(st, dict) and hasattr(st, "to_dict"):
                try:
                    st = st.to_dict()
                except Exception:
                    st = {}
            if not isinstance(st, dict):
                st = {}

            # pick orders (store first, set later)
            tmp_orders: Dict[str, List[str]] = {}
            all_infos: Dict[str, Tuple[List[Tuple[Any, float]], str, List[Tuple[Any, float, str]], List[Tuple[Any, float]]]] = {}

            per_power_time: Dict[str, float] = {}

            # 先取当前单位信息；无单位的国家直接跳过选单，更省算力
            units = st.get("units", {}) or {}

            choose_all_t0 = time.perf_counter()
            for pwr in POWERS:
                # 当前国家没有 unit：直接跳过，不调用 agent
                if len(units.get(pwr) or []) == 0:
                    tmp_orders[pwr] = []
                    all_infos[pwr] = ([], "skip_no_units", [], [])
                    per_power_time[pwr] = 0.0
                    continue

                ag = agent_for_power[pwr]
                pwr_t0 = time.perf_counter()
                orders, items, used_source, dropped, raw_items = choose_orders_wrapper(
                    ag,
                    game=game,
                    power=pwr,
                    state=states[pwr],
                    source=args.source,
                    top_k=args.topk,
                    mode=args.mode,
                )
                pwr_elapsed = time.perf_counter() - pwr_t0

                tmp_orders[pwr] = orders
                all_infos[pwr] = (items, used_source, dropped, raw_items)
                per_power_time[pwr] = pwr_elapsed
            choose_all_elapsed = time.perf_counter() - choose_all_t0

            f.write("\n" + "=" * 90 + "\n")
            f.write(f"[STEP {step:04d}] phase={phase}\n")

            # units = st.get("units", {}) or {}
            centers = st.get("centers", {}) or {}
            influence = st.get("influence", None)
            terr_src = "influence" if isinstance(influence, dict) else "fallback"

            f.write(f"[STATE BEFORE] terr_src={terr_src}\n")
            for pwr in POWERS:
                ulist = list((units.get(pwr) or []))
                sc_set, unit_set, past_free_set = get_territory_parts(st, pwr)
                terr_set = sc_set | unit_set | past_free_set
                sc_list = sorted(sc_set)
                non_sc = sorted(terr_set - sc_set)
                f.write(
                    f"  {pwr}: "
                    f"units({len(ulist)})={ulist} | "
                    f"SC({len(sc_list)})={sc_list} | "
                    f"nonSC({len(non_sc)})={non_sc}\n"
                )

            # per-power info
            for pwr in POWERS:
                items, used_source, dropped, raw_items = all_infos[pwr]
                tagp = _tag(pwr)

                f.write(
                    f"[AGENT] power={pwr} tag={tagp} used_source={used_source} "
                    f"raw_n={len(raw_items)} kept_n={len(items)} dropped_n={len(dropped)} "
                    f"choose_time={per_power_time[pwr]:.4f}s\n"
                )

                if tagp == "consistent":
                    f.write(f"[FINAL CHOICE] power={pwr} orders={tmp_orders.get(pwr, [])}\n")

                    # 无单位时前面已经直接跳过选单了，这里不要再做异常检查
                    if used_source != "skip_no_units":
                        _check_consistent_candidate_bug_or_raise(
                            f=f,
                            phase_str=phase_str,
                            power=pwr,
                            raw_items=raw_items,
                            items=items,
                            dropped=dropped,
                            orders=tmp_orders.get(pwr, []),
                        )

            # set orders
            set_t0 = time.perf_counter()
            f.write("[ORDERS SET]\n")
            for pwr in POWERS:
                game.set_orders(pwr, tmp_orders.get(pwr, []))
                f.write(f"  {pwr}: {tmp_orders.get(pwr, [])}\n")
            set_elapsed = time.perf_counter() - set_t0

            f.write("[POST CHECK]\n")

            checker = agent_pool.get("consistent", None)
            if checker is None or (not hasattr(checker, "audit_final_orders")):
                checker = None

            for pwr in POWERS:
                orders = tmp_orders.get(pwr, [])

                if checker is None:
                    per_power_check_counts[pwr]["NONE"] += 1
                    total_check_counts["NONE"] += 1
                    f.write(f"  {pwr}: NONE\n")
                    continue

                rep = checker.audit_final_orders(game=game, power=pwr, orders=orders)
                violations = rep.get("violations", []) or []

                if not violations:
                    per_power_check_counts[pwr]["NONE"] += 1
                    total_check_counts["NONE"] += 1
                    f.write(f"  {pwr}: NONE\n")
                else:
                    # ✅ 同一条 orders 可能同时触发多个 check：都记录
                    order_list = ["C1", "C2", "C3", "C4"]
                    tagset = {ctag for ctag, _ in violations}
                    tags = [t for t in order_list if t in tagset]

                    for t in tags:
                        per_power_check_counts[pwr][t] += 1
                        total_check_counts[t] += 1

                    f.write(f"  {pwr}: {_fmt_post_check_violations(violations)}\n")

            process_t0 = time.perf_counter()
            try:
                game.process()
            except Exception as e:
                process_elapsed = time.perf_counter() - process_t0
                phase_elapsed = time.perf_counter() - phase_t0
                f.write(
                    f"[TIMING] choose_all={choose_all_elapsed:.4f}s set_orders={set_elapsed:.4f}s "
                    f"process={process_elapsed:.4f}s phase_total={phase_elapsed:.4f}s\n"
                )
                f.write(f"[ERROR] game.process() failed @phase={phase_str}: {repr(e)}\n")
                run_error = (phase_str, repr(e))
                end_reason = "exception"
                break

            process_elapsed = time.perf_counter() - process_t0
            phase_elapsed = time.perf_counter() - phase_t0
            f.write(
                f"[TIMING] choose_all={choose_all_elapsed:.4f}s "
                f"set_orders={set_elapsed:.4f}s "
                f"process={process_elapsed:.4f}s "
                f"phase_total={phase_elapsed:.4f}s\n"
            )

            # ----------------------------
            # ✅ support_total / support_success（严格匹配版）
            #   success 定义：
            #     - support move：被支持方自己确实下了 A - B
            #     - support hold：被支持方自己确实下了 A H
            #   注意：这里只看“支持对象是否对上”，不看最终是否打赢/占住
            # ----------------------------
            st_after = game.get_state()
            if not isinstance(st_after, dict) and hasattr(st_after, "to_dict"):
                try:
                    st_after = st_after.to_dict()
                except Exception:
                    st_after = {}
            if not isinstance(st_after, dict):
                st_after = {}

            # st_before：用于根据被支持单位初始位置，反查它属于哪个国家
            units_before = (st.get("units", {}) or {})
            occ = {}
            for pwr0, ulist0 in units_before.items():
                for u0 in (ulist0 or []):
                    parts0 = str(u0).strip().lstrip("*").split()
                    if len(parts0) >= 2:
                        occ[parts0[1]] = str(pwr0)

            def _same_loc(a: str, b: str) -> bool:
                """位置匹配：完全相同，或一方省略 coast 时按主地区匹配"""
                a = str(a)
                b = str(b)
                return a == b or a.split("/")[0] == b.split("/")[0]

            def _find_owner_by_loc(loc: str):
                """根据初始位置反查该单位属于哪个国家"""
                if loc in occ:
                    return occ[loc]

                base = str(loc).split("/")[0]
                cand = [v for k, v in occ.items() if k.split("/")[0] == base]
                if cand and all(x == cand[0] for x in cand):
                    return cand[0]

                return None

            def _is_exact_supported_move(order: str, src: str, dest: str) -> bool:
                """判断某条 order 是否正好是 src -> dest"""
                toks = str(order).strip().split()
                if len(toks) < 4:
                    return False
                if toks[2] != "-":
                    return False
                return _same_loc(toks[1], src) and _same_loc(toks[3], dest)

            def _is_exact_supported_hold(order: str, src: str) -> bool:
                """判断某条 order 是否正好是 src H"""
                toks = str(order).strip().split()
                if len(toks) < 3:
                    return False
                if toks[2] != "H":
                    return False
                return _same_loc(toks[1], src)

            for pwr in POWERS:
                for od in (tmp_orders.get(pwr, []) or []):
                    s = str(od)
                    if " S " not in s:
                        continue

                    # 该玩家发出了一条 support 命令
                    support_total[pwr] += 1

                    rhs = s.split(" S ", 1)[1].strip()
                    rtoks = rhs.split()
                    if len(rtoks) < 2:
                        continue

                    # 被支持单位的初始位置
                    sup_loc = rtoks[1]
                    sup_owner = _find_owner_by_loc(sup_loc)
                    if not sup_owner:
                        continue

                    # 被支持玩家这回合自己实际下的 orders
                    supported_orders = tmp_orders.get(sup_owner, []) or []

                    # support move：检查被支持玩家是否真的下了 src -> dest
                    if "-" in rtoks:
                        i = rtoks.index("-")
                        if i + 1 < len(rtoks):
                            dest = rtoks[i + 1]
                            if any(_is_exact_supported_move(x, sup_loc, dest) for x in supported_orders):
                                support_success[pwr] += 1

                    # support hold：检查被支持玩家是否真的下了 src H
                    else:
                        if any(_is_exact_supported_hold(x, sup_loc) for x in supported_orders):
                            support_success[pwr] += 1

            step += 1 
            # ----------------------------
            # ✅ 终止条件 1：1913 年结算后立即终止（处理完 W1913A）
            # ----------------------------
            if phase_str == STOP_AFTER_PHASE:
                end_reason = "stop_after_W1913A"
                f.write(f"[TERMINATION] reason={end_reason}\n")
                break

            # ----------------------------
            # ✅ 终止条件 2：引擎已结束（可能是 18 SC / draw / 其他）
            # ----------------------------
            if is_game_done(game):
                # 直接从 st_after centers 看是否有人 >=18
                centers = (st_after.get("centers", {}) or {})
                winner = None
                for p in POWERS:
                    if len(centers.get(p, []) or []) >= 18:
                        winner = p
                        break
                end_reason = f"solo_18sc_{winner}" if winner else "engine_done"
                f.write(f"[TERMINATION] reason={end_reason}\n")
                break

            f.flush()
            # step += 1


        f.write("\n[CHECK COUNTS PER POWER]\n")
        for pwr in POWERS:
            c = per_power_check_counts[pwr]
            f.write(
                f"  {pwr}: "
                f"C1={c['C1']} C2={c['C2']} C3={c['C3']} C4={c['C4']} NONE={c['NONE']}\n"
            )

        f.write("[CHECK COUNTS TOTAL]\n")
        f.write(
            f"  C1={total_check_counts['C1']} "
            f"C2={total_check_counts['C2']} "
            f"C3={total_check_counts['C3']} "
            f"C4={total_check_counts['C4']} "
            f"NONE={total_check_counts['NONE']}\n"
        )  
        # 兜底 end_reason
        if end_reason is None:
            if run_error is not None:
                end_reason = "exception"
            elif step >= args.max_phases:
                end_reason = "max_phases"
            elif is_game_done(game):
                end_reason = "engine_done"
            else:
                end_reason = "stopped_unknown"

        final_state = game.get_state()
        if not isinstance(final_state, dict) and hasattr(final_state, "to_dict"):
            try:
                final_state = final_state.to_dict()
            except Exception:
                final_state = {}
        if not isinstance(final_state, dict):
            final_state = {}

        # SC counts 直接取 centers（你说的“引擎里调就行”）
        final_centers = final_state.get("centers", {}) or {}
        sc_counts = {p: len(final_centers.get(p, []) or []) for p in POWERS}

        # SoS 用你给的函数
        sos_list = compute_game_sos_from_state(final_state)  # List[float] 与 POWERS 对齐
        sos_map = {p: float(sos_list[i]) for i, p in enumerate(POWERS)}

        final_phase = last_processed_phase or str(game.get_current_phase())

        # === FINAL SUMMARY ===
        f.write("\n=== FINAL SUMMARY ===\n")
        f.write(f"game_id={game_id}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"setup={args.setup}\n")
        f.write(f"my_power={args.power}\n")
        f.write(f"my_agent={args.my_agent}\n")
        f.write(f"opp_agent={args.opp_agent}\n")
        f.write(f"end_reason={end_reason}\n")
        f.write(f"final_phase={final_phase}\n")
        f.write(f"num_phases={step}\n")

        f.write("sos_each_power=" + ",".join([f"{p}:{sos_map[p]:.6f}" for p in POWERS]) + "\n")

        f.write("c1_count_each_power=" + ",".join([f"{p}:{per_power_check_counts[p]['C1']}" for p in POWERS]) + "\n")
        f.write("c2_count_each_power=" + ",".join([f"{p}:{per_power_check_counts[p]['C2']}" for p in POWERS]) + "\n")
        f.write("c3_count_each_power=" + ",".join([f"{p}:{per_power_check_counts[p]['C3']}" for p in POWERS]) + "\n")
        f.write("c4_count_each_power=" + ",".join([f"{p}:{per_power_check_counts[p]['C4']}" for p in POWERS]) + "\n")

        f.write("support_success_each_power=" + ",".join([f"{p}:{support_success[p]}" for p in POWERS]) + "\n")
        f.write("support_total_each_power=" + ",".join([f"{p}:{support_total[p]}" for p in POWERS]) + "\n")

        f.write("\n[FINAL SC]\n")
        for p in POWERS:
            f.write(f"{p}={sc_counts[p]}\n")

        f.write("\n[CONSISTENCY COUNTS]\n")
        for p in POWERS:
            c = per_power_check_counts[p]
            f.write(f"{p}: C1={c['C1']} C2={c['C2']} C3={c['C3']} C4={c['C4']}\n") 
        f.write("\n=== RUN END ===\n")
        f.write(f"final_phase={game.get_current_phase()}\n")
        row = {
            "game_id": game_id,
            "seed": args.seed,
            "setup": args.setup,
            "my_power": args.power,
            "my_agent": args.my_agent,
            "opp_agent": args.opp_agent,
            "end_reason": end_reason,
            "final_phase": final_phase,
            "num_phases": step,
        }

        for p in POWERS:
            row[f"sc_{p}"] = sc_counts[p]
        for p in POWERS:
            row[f"sos_{p}"] = sos_map[p]
        for p in POWERS:
            row[f"c1_{p}"] = per_power_check_counts[p]["C1"]
            row[f"c2_{p}"] = per_power_check_counts[p]["C2"]
            row[f"c3_{p}"] = per_power_check_counts[p]["C3"]
            row[f"c4_{p}"] = per_power_check_counts[p]["C4"]
        for p in POWERS:
            row[f"support_success_{p}"] = support_success[p]
            row[f"support_total_{p}"] = support_total[p]

        need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        with open(csv_path, "a", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=list(row.keys()))
            if need_header:
                w.writeheader()
            w.writerow(row)

        f.write(f"\n[CSV SAVED] {csv_path}\n")
        f.flush()

    print(f"[OK] log saved to: {log_path}")


if __name__ == "__main__":
    main()