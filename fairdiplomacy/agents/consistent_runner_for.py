# coding=utf-8
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import heyhi

# runner 在 fairdiplomacy/agents 下时，优先相对导入
try:
    from .consistent_agent import POWERS, get_territory_parts, load_cicero
except Exception:
    from fairdiplomacy.agents.consistent_agent import POWERS, get_territory_parts, load_cicero
import heyhi
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent
from fairdiplomacy.agents.searchbot_agent import SearchBotAgent
from fairdiplomacy.agents.base_strategy_model_agent import BaseStrategyModelAgent
import time

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
AGENT_KINDS: List[AgentKind] = ["consistent", "cicero_nopress", "diplodocus_high", "searchbot", "dipnet"]


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
) -> Tuple[List[str], List[Tuple[Any, float]], str, List[Tuple[Any, float, str]]]:
    """
    - ConsistentAgent: 用 choose_orders（保留你的过滤/候选输出）
    - 其它对手：统一 get_orders
    """
    if hasattr(agent, "choose_orders"):
        return agent.choose_orders(
            game=game,
            power=power,
            agent_state=state,
            source=source,
            top_k=top_k,
            mode=mode,
        )

    orders = agent.get_orders(game, power=power, state=state)
    return orders, [], type(agent).__name__, []


# ----------------------------
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
    parser.add_argument("--cfg_searchbot", type=str, default="conf/common/agents/searchbot.prototxt")
    parser.add_argument("--cfg_dipnet", type=str, default="conf/common/agents/base_strategy_model.prototxt")

    parser.add_argument("--project_root", type=str, default="/workspace/Diplomacy/diplomacy_cicero")
    parser.add_argument("--power", type=str, default="AUSTRIA", choices=POWERS)
    parser.add_argument("--setup", type=str, default="1v6", choices=["1v6", "all7"])
    parser.add_argument("--seed", type=int, default=0)

    # ✅ 策略选择：包含我们自己 + 4 个对手
    # 默认先测 DipNet：opp_agent=dipnet
    AGENT_TYPES = ["consistent", "cicero_nopress", "diplodocus_high", "searchbot", "dipnet"]
    parser.add_argument("--my_agent", type=str, default="consistent", choices=AGENT_TYPES)
    parser.add_argument("--opp_agent", type=str, default="dipnet", choices=AGENT_TYPES)
    parser.add_argument("--all_agent", type=str, default="consistent", choices=AGENT_TYPES)
        # only affects consistent branch
    parser.add_argument("--source", type=str, default="bqre_topK", choices=["bqre_topK", "search_br", "bp"])
    parser.add_argument("--mode", type=str, default="top1", choices=["top1", "sample"])
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
    default_name = f"run_{args.setup}_my{args.power}_my{args.my_agent}_opp{args.opp_agent}_all{args.all_agent}_{ts}.log"
    log_path = args.log if args.log else os.path.join(log_dir, default_name)

    # ---- decide which agent kinds are actually needed in THIS run ----
    needed: set[str] = set()
    if args.setup == "all7":
        needed.add(args.all_agent)
    else:
        needed.add(args.my_agent)
        needed.add(args.opp_agent)

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
            return load_cicero(args.cfg_consistent, skip_cache=False)

        if t == "cicero_nopress":
            return load_cicero_nopress_agent(args.cfg_cicero_nopress, skip_cache=False)

        if t == "diplodocus_high":
            return load_diplodocus_high_agent(args.cfg_diplodocus_high, skip_cache=False)

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

    # init states by their own agent
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
        f.write(f"cfg_searchbot={args.cfg_searchbot}\n")
        f.write(f"cfg_dipnet={args.cfg_dipnet}\n")
        f.write(f"my_agent={args.my_agent} opp_agent={args.opp_agent} all_agent={args.all_agent}\n")
        f.write(f"seed={args.seed} source={args.source} mode={args.mode} topk={args.topk} max_phases={args.max_phases}\n")

        f.write("[ASSIGNMENT]\n")
        for p in POWERS:
            f.write(f"  {p}: {_tag(p)}\n")
        f.write("\n")
        f.flush()

        step = 0

        while step < args.max_phases and not is_game_done(game):
            phase_t0 = time.perf_counter()
            phase = game.get_current_phase()

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
            all_infos: Dict[str, Tuple[List[Tuple[Any, float]], str, List[Tuple[Any, float, str]]]] = {}
            per_power_time: Dict[str, float] = {}

            choose_all_t0 = time.perf_counter()
            for pwr in POWERS:
                ag = agent_for_power[pwr]

                pwr_t0 = time.perf_counter()
                orders, items, used_source, dropped = choose_orders_wrapper(
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
                all_infos[pwr] = (items, used_source, dropped)
                per_power_time[pwr] = pwr_elapsed
            choose_all_elapsed = time.perf_counter() - choose_all_t0

            f.write("\n" + "=" * 90 + "\n")
            f.write(f"[STEP {step:04d}] phase={phase}\n")

            units = st.get("units", {}) or {}
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

            # per-power info (consistent has dropped; others will be empty)
            for pwr in POWERS:
                items, used_source, dropped = all_infos[pwr]
                f.write(
                    f"[AGENT] power={pwr} tag={_tag(pwr)} used_source={used_source} "
                    f"topk={len(items)} choose_time={per_power_time[pwr]:.4f}s\n"
                )

                # 可选：如果你想看“offer set最终输出/最终采用动作”
                if items:
                    f.write(f"[TOP CANDIDATES] power={pwr} n={len(items)}\n")
                    for j, (a, pp) in enumerate(items):
                        act_str = "[" + ", ".join(map(str, a)) + "]" if isinstance(a, (list, tuple)) else str(a)
                        f.write(f"  +{j:02d}  p={float(pp):.8f}  action={act_str}\n")

                f.write(f"[FILTERED OUT] power={pwr} n={len(dropped)}\n")
                for j, (a, pp, rsn) in enumerate(dropped):
                    act_str = "[" + ", ".join(map(str, a)) + "]" if isinstance(a, (list, tuple)) else str(a)
                    f.write(f"  -{j:02d}  p={float(pp):.8f}  reason={rsn}  action={act_str}\n")

            # set orders
            set_t0 = time.perf_counter()
            f.write("[ORDERS SET]\n")
            for pwr in POWERS:
                game.set_orders(pwr, tmp_orders.get(pwr, []))
                f.write(f"  {pwr}: {tmp_orders.get(pwr, [])}\n")
            set_elapsed = time.perf_counter() - set_t0

            # process
            process_t0 = time.perf_counter()
            try:
                game.process()
            except Exception as e:
                process_elapsed = time.perf_counter() - process_t0
                phase_elapsed = time.perf_counter() - phase_t0
                f.write(f"[TIMING] choose_all={choose_all_elapsed:.4f}s set_orders={set_elapsed:.4f}s "
                        f"process={process_elapsed:.4f}s phase_total={phase_elapsed:.4f}s\n")
                f.write(f"[ERROR] game.process() failed @phase={phase}: {repr(e)}\n")
                break
            process_elapsed = time.perf_counter() - process_t0

            phase_elapsed = time.perf_counter() - phase_t0
            f.write(
                f"[TIMING] choose_all={choose_all_elapsed:.4f}s "
                f"set_orders={set_elapsed:.4f}s "
                f"process={process_elapsed:.4f}s "
                f"phase_total={phase_elapsed:.4f}s\n"
            )

            f.flush()
            step += 1
        
        f.write("\n=== RUN END ===\n")
        f.write(f"final_phase={game.get_current_phase()}\n")
        f.flush()

    print(f"[OK] log saved to: {log_path}")


if __name__ == "__main__":
    main()