# coding=utf-8
"""
coo_agent.py - 修复版本 + 封装 Cicero 加载

load方法借鉴：
https://github.com/ALLAN-DIP/diplomacy_cicero/blob/deception_friction_value/fairdiplomacy_external/friction/utils.py#L492

def load_cicero()

https://github.com/KaiXIIM/dipllm/tree/main/conf
https://github.com/KaiXIIM/dipllm/blob/main/fairdiplomacy/agents/llm_agent.py


"""

from typing import Any, Dict, List, Tuple
import logging
import os

from fairdiplomacy import pydipcc
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent
from collections import defaultdict
LOG = logging.getLogger(__name__)

def dump_legal_support_orders(game, *, max_each_unit=10):
    """
    直接从 dipcc 规则引擎拿“当前局面每个单位的所有合法 orders”，看 support 是否存在。
    """
    all_poss = game.get_all_possible_orders()  # dict: loc -> List[str]
    support_by_loc = defaultdict(list)

    for loc, orders in all_poss.items():
        supp = [o for o in orders if " S " in o]  # DipCC support 通常是空格包围的 S
        if supp:
            support_by_loc[loc] = supp[:max_each_unit]

    print("\n" + "=" * 80)
    print("[LEGAL CHECK] 当前局面 dipcc 合法 orders 中的 SUPPORT 列表（按单位loc）")
    if not support_by_loc:
        print("  => 没有任何合法 support（这通常意味着不是 Movement phase 或局面不允许）")
    else:
        for loc, supp in support_by_loc.items():
            print(f"  - {loc}:")
            for s in supp:
                print(f"      {s}")
    print("=" * 80 + "\n")

    return support_by_loc
def _get_power_unit_locs(game: "pydipcc.Game", power: str) -> List[str]:
    st = game.get_state()
    # st["units"][power] 形如 ["A PAR", "A MAR", "F BRE"] 或带海岸 "F STP/SC"
    units = st["units"].get(power, [])
    locs = []
    for u in units:
        parts = u.split()
        if len(parts) >= 2:
            locs.append(parts[1])
    return locs

def dump_power_legal_support_orders(game, power: str, *, max_each_unit=50):
    all_poss = game.get_all_possible_orders()
    locs = _get_power_unit_locs(game, power)

    print("\n" + "=" * 80)
    print(f"[LEGAL CHECK - {power}] 仅该势力单位的 SUPPORT orders")
    if not locs:
        print(f"  => state 里没找到 {power} 的单位？")
        print("=" * 80 + "\n")
        return {}

    out = {}
    for loc in locs:
        orders = all_poss.get(loc, [])
        supp = [o for o in orders if " S " in o]
        if supp:
            out[loc] = supp[:max_each_unit]

    if not out:
        print("  => 该势力单位没有任何合法 support（可能是 phase 或局面原因）")
    else:
        for loc, supp in out.items():
            print(f"  - {loc}:")
            for s in supp:
                print(f"      {s}")
    print("=" * 80 + "\n")
    return out
def support_stats(action_to_prob, token=" S "):
    total = len(action_to_prob)
    sup_actions = [(a, p) for a, p in action_to_prob.items() if any(token in o for o in a)]
    sup_cnt = len(sup_actions)
    sup_mass = sum(p for _, p in sup_actions)
    sup_actions.sort(key=lambda x: x[1], reverse=True)
    return total, sup_cnt, sup_mass, sup_actions[:10]

POWERS = ["AUSTRIA","ENGLAND","FRANCE","GERMANY","ITALY","RUSSIA","TURKEY"]

import random
from typing import Optional

def _normalize_dist(items: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    s = sum(max(0.0, p) for _, p in items)
    if s <= 0:
        return [(a, 1.0 / len(items)) for a, _ in items] if items else []
    return [(a, max(0.0, p) / s) for a, p in items]

def _sample_from_items(items: List[Tuple[Any, float]]) -> Any:
    # items: [(action, prob), ...] probs 已归一化
    r = random.random()
    cum = 0.0
    for a, p in items:
        cum += p
        if r <= cum:
            return a
    return items[-1][0]  # 数值误差兜底

def get_orderable_locs(game: "pydipcc.Game", power: str) -> List[str]:
    """
    兼容不同 pydipcc 版本：
    - 新版：game.get_orderable_locations() -> Dict[power, List[loc]]
    - 若 state 里带 orderable_locations 也可读
    - 再不行：fallback 到 units 的 loc（仅适用于 Movement 相位）
    """
    # 1) 优先用 API（你当前就是这种：无参 -> dict）
    if hasattr(game, "get_orderable_locations"):
        try:
            res = game.get_orderable_locations()   # ✅ 不传 power
            if isinstance(res, dict):
                return res.get(power, []) or []
        except TypeError:
            # 极少数旧版可能是带参的，这里兜底
            try:
                res = game.get_orderable_locations(power)
                if isinstance(res, list):
                    return res
            except Exception:
                pass
        except Exception:
            pass

    # 2) 再从 state 里找（你当前 state keys 没有这个，但保留兼容）
    st = game.get_state()
    ol = st.get("orderable_locations", None)
    if isinstance(ol, dict):
        return ol.get(power, []) or []

    # 3) 最后 fallback：用 units 推 loc（Movement phase 有效）
    locs = []
    for u in (st.get("units", {}).get(power, []) or []):
        parts = u.split()
        if len(parts) >= 2:
            locs.append(parts[1])
    return locs


def pick_random_legal_orders(game: "pydipcc.Game", power: str) -> List[str]:
    """
    给其他势力用：每个 orderable loc 随机挑一个合法 order。
    若某些 phase 没有 orderable loc，就返回空列表（让 dipcc 默认处理）。
    """
    all_poss = game.get_all_possible_orders()
    locs = get_orderable_locs(game, power)

    orders = []
    for loc in locs:
        cand = all_poss.get(loc, [])
        if cand:
            orders.append(random.choice(cand))
    return orders

def parse_unit(u: str):
    # "A PAR" / "F STP/SC"
    parts = u.split()
    if len(parts) < 2:
        return ("?", u)
    return (parts[0], parts[1])

def dump_current_assets(game: "pydipcc.Game"):
    st = game.get_state()
    phase = game.get_current_phase()
    print("\n" + "=" * 90)
    print(f"[CURRENT STATE] phase={phase}")
    print("state keys =", list(st.keys()))
    print("=" * 90)

    units = st.get("units", {})      # Dict[power, List[str]]
    centers = st.get("centers", {})  # Dict[power, List[str]]
    influence = st.get("influence", None)  # Dict[power, List[str]]  (不一定存在)

    for p in POWERS:
        us = units.get(p, []) or []
        cs = centers.get(p, []) or []
        inf = (influence.get(p, []) if isinstance(influence, dict) else None)

        unit_cnt = len(us)
        sc_cnt = len(cs)
        net_build = sc_cnt - unit_cnt  # >0 表示“理论上可建造”，<0 表示“理论上需裁军”（只在冬季调整相位生效）

        print(f"\n--- {p} ---")
        print(f"Units({unit_cnt}): " + ", ".join([f"{parse_unit(x)[0]}@{parse_unit(x)[1]}" for x in us]) if us else "Units(0)")
        print(f"Supply Centers({sc_cnt}): " + ", ".join(cs) if cs else "Supply Centers(0)")
        print(f"Net builds (SC - Units): {net_build:+d}")

        if inf is None:
            print("Influence: <not provided by this state>")
        else:
            print(f"Influence({len(inf)}): " + ", ".join(inf))

    print("\n" + "=" * 90 + "\n")
def get_phase_history_safe(game: "pydipcc.Game"):
    if hasattr(game, "get_phase_history"):
        return game.get_phase_history()
    raise RuntimeError("This pydipcc.Game does not expose get_phase_history()")

from collections.abc import Mapping

def _phase_name(ph) -> str:
    # dict 版本
    if isinstance(ph, Mapping):
        return ph.get("name", "<no-name>")
    # PhaseData / pybind 版本
    for attr in ("name", "phase", "short_phase"):
        if hasattr(ph, attr):
            try:
                v = getattr(ph, attr)
                return str(v() if callable(v) else v)
            except Exception:
                pass
    return str(ph)

def _phase_orders(ph):
    # dict 版本
    if isinstance(ph, Mapping):
        return ph.get("orders", {}) or {}

    # PhaseData / pybind 版本：常见是 ph.orders / ph.get_orders()
    for attr in ("orders", "get_orders"):
        if hasattr(ph, attr):
            try:
                v = getattr(ph, attr)
                v = v() if callable(v) else v
                # v 可能是 dict，也可能是 pybind map，尽量转成 python dict
                if isinstance(v, Mapping):
                    return dict(v)
                try:
                    return dict(v)
                except Exception:
                    return v
            except Exception:
                pass
    return {}

def dump_order_history(game: "pydipcc.Game", *, last_n: int = 9999, only_powers=None):
    hist = get_phase_history_safe(game)
    if not hist:
        print("[HISTORY] phase_history is empty (no processed phases yet?)")
        return

    only_powers = set(only_powers) if only_powers else None
    hist = hist[-last_n:]

    print("\n" + "=" * 90)
    print(f"[ORDER HISTORY] phases={len(hist)} (show last_n={last_n})")
    print("=" * 90)

    for ph in hist:
        name = _phase_name(ph)
        orders = _phase_orders(ph) or {}
        print(f"\n### {name} ###")

        if not orders:
            print("  (no orders recorded)")
            continue

        for p in POWERS:
            if only_powers and p not in only_powers:
                continue
            olist = orders.get(p, None) if isinstance(orders, dict) else None
            if olist is None:
                continue

            if isinstance(olist, (list, tuple)):
                print(f"  {p}: {', '.join(olist) if olist else '<empty>'}")
            else:
                print(f"  {p}: {olist}")

    print("\n" + "=" * 90 + "\n")

class CooAgent(BQRE1PAgent):
    """
    Cicero 价值查看器 Agent
    """

    def __init__(self, cfg, **kwargs) -> None:
        # cfg 里已经包含 base_strategy_model { model_path: "...", temperature: ... }
        # 父类会用它自动加载 blueprint 模型
        super().__init__(cfg, **kwargs)
        LOG.info("[CooAgent] 初始化完成，已加载 Cicero 价值模型和蓝图策略")
    def get_action_candidates(
        self,
        game: "pydipcc.Game",
        agent_power: str,
        *,
        agent_state: Any = None,
        source: str = "bp",     # "bp" or "search_br"
        top_k: int = 30,
        verbose: bool = True,
    ) -> List[Tuple[Any, float]]:
        """
        返回“可选 joint actions”的(动作, 概率)列表。
        - source="bp": 直接用 blueprint 的 plausible joint actions（动作多）
        - source="search_br": 用 BR 搜索结果的 agent_policy（可能只剩 1 个）
        """
        current_phase = game.get_current_phase()
        if agent_state is None:
            agent_state = self.initialize_state(agent_power)

        # 1) BP plausible policy（通常动作多）
        bp_policy: Dict[str, Dict[Any, float]] = self.get_plausible_orders_policy(
            game=game,
            agent_power=agent_power,
            agent_state=agent_state,
        )
        if agent_power not in bp_policy:
            if verbose:
                print(f"[get_action_candidates] No policy for {agent_power} at {current_phase}")
            return []

        dist = bp_policy[agent_power]

        # 2) 可选：用 BR 搜索覆盖（可能坍缩为 1）
        if source == "search_br":
            try:
                search_result = self.run_best_response_against_correlated_bilateral_search(
                    game=game,
                    agent_power=agent_power,
                    bp_policy=bp_policy,
                    agent_state=agent_state,
                )
                agent_policies = search_result.get_agent_policy()
                if agent_power in agent_policies and agent_policies[agent_power]:
                    dist = agent_policies[agent_power]
            except Exception as e:
                if verbose:
                    print(f"[get_action_candidates] search_br failed: {e}. fallback to bp.")

        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None and top_k > 0:
            items = items[:top_k]

        # 统计 support 占比（沿用你写的 support_stats）
        if verbose and items:
            tot = len(dist)
            sub = dict(items)
            t2, sup_cnt, sup_mass, top10 = support_stats(sub)
            print(f"[CAND] {agent_power} phase={current_phase} source={source} top_k={len(items)} / total={tot}")
            print(f"       support joint actions = {sup_cnt}/{len(items)}, prob_mass={sup_mass:.6f}")
            for a, p in top10:
                print(f"         p={p:.6f} | {a}")

        return items

    def choose_action(
        self,
        game: "pydipcc.Game",
        agent_power: str,
        *,
        agent_state: Any = None,
        source: str = "bp",
        top_k: int = 30,
        mode: str = "sample",   # "sample" or "top1"
        verbose: bool = False,
    ) -> Tuple[List[str], Any]:
        """
        选一个 joint action，并返回:
        - orders_list: List[str] 传给 game.set_orders
        - action: 原始 joint action (tuple[str,...])
        """
        items = self.get_action_candidates(
            game, agent_power, agent_state=agent_state,
            source=source, top_k=top_k, verbose=verbose
        )
        if not items:
            return [], None

        if mode == "top1":
            action = items[0][0]
        else:
            normed = _normalize_dist(items)
            action = _sample_from_items(normed)

        orders = list(action) if isinstance(action, (list, tuple)) else [action]
        return orders, action
    def evaluate_action_values(
        self,
        game: "pydipcc.Game",
        agent_power: str,
        *,
        agent_state: Any = None,
    ) -> List[Tuple[Any, float]]:
        """计算当前可选动作的价值"""

        current_phase = game.get_current_phase()

        # ✅ 关键修复：如果没有agent_state，先初始化一个
        if agent_state is None:
            agent_state = self.initialize_state(agent_power)
            LOG.info(f"[CooAgent] 为 {agent_power} 初始化了新的 agent_state")

        # 1. 获取蓝图策略
        bp_policy: Dict[str, Dict[Any, float]] = self.get_plausible_orders_policy(
            game=game,
            agent_power=agent_power,
            agent_state=agent_state,
        )

        if agent_power not in bp_policy:
            LOG.warning(
                "[CooAgent] Phase %s 没有可用动作给 %s",
                current_phase,
                agent_power,
            )
            return []

        our_bp_policy = bp_policy[agent_power]
        tot, sup_cnt, sup_mass, top10 = support_stats(our_bp_policy)
        print(f"[BP] support joint actions = {sup_cnt}/{tot}, prob_mass={sup_mass:.6f}")
        for a, p in top10:
            print(f"  bp p={p:.6f} | {a}")

        LOG.info(
            "[CooAgent] Phase %s, Power %s, BP策略中有 %d 个可行动作",
            current_phase,
            agent_power,
            len(our_bp_policy),
        )

        # 2. 运行 Cicero 搜索获取价值
        search_result = self.run_best_response_against_correlated_bilateral_search(
            game=game,
            agent_power=agent_power,
            bp_policy=bp_policy,
            agent_state=agent_state,  # ✅ 现在有了 agent_state
        )

        # 3. 提取搜索结果
        agent_policies: Dict[str, Dict[Any, float]] = search_result.get_agent_policy()
        if agent_power not in agent_policies:
            LOG.warning("[CooAgent] 搜索结果中没有 %s 的策略", agent_power)
            return []

        our_search_policy: Dict[Any, float] = agent_policies[agent_power]
        action_values: List[Tuple[Any, float]] = sorted(
            our_search_policy.items(), key=lambda kv: kv[1], reverse=True
        )
        tot, sup_cnt, sup_mass, top10 = support_stats(our_search_policy)
        print(f"[SEARCH] support joint actions = {sup_cnt}/{tot}, prob_mass={sup_mass:.6f}")
        for a, p in top10:
            print(f"  se p={p:.6f} | {a}")


        # 4. 打印结果
        print(f"\n{'=' * 70}")
        print(f"[CooAgent] Cicero动作价值分析")
        print(f"  Power: {agent_power}")
        print(f"  Phase: {current_phase}")
        print(f"{'=' * 70}")
        for idx, (action, value) in enumerate(action_values):
            print(f"  [{idx:2d}] 价值={value:.4f} | 动作={action}")
        print(f"{'=' * 70}\n")

        return action_values
import heyhi  # 放在文件顶部一起 import 就行


def load_cicero(
    cfg_path: str = "conf/common/agents/coo_agent.prototxt",
    *,
    skip_cache: bool = False,
    **agent_kwargs,
) -> CooAgent:
    """
    一步完成：
    1）从 prototxt 读 Cicero 的 agent 配置
    2）取出 bqre1p 子配置（包含 base_strategy_model）
    3）构造 CooAgent 实例（内部会加载 blueprint.pt）
    """
    LOG.info("[load_cicero] 从 %s 读取配置", cfg_path)
    full_cfg = heyhi.load_config(cfg_path)

    # Cicero 配置存在两种写法：agent { bqre1p { ... } } 或直接 bqre1p { ... }
    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "coo_agent"):
        agent_cfg = full_cfg.agent.coo_agent
        LOG.info("[load_cicero] 使用 full_cfg.agent.coo_agent")
    elif hasattr(full_cfg, "coo_agent"):
        agent_cfg = full_cfg.coo_agent
        LOG.info("[load_cicero] 使用 full_cfg.coo_agent")
    else:
        raise ValueError(
            "配置文件格式错误！期望结构: agent.bqre1p 或 bqre1p\n"
            f"实际可用字段: {dir(full_cfg)}"
        )

    # 这里如果你想检查/修改 base_strategy_model，也可以：
    # LOG.info("Blueprint 模型路径: %s", agent_cfg.base_strategy_model.model_path)
    # agent_cfg.base_strategy_model.model_path = "models/your_own_blueprint.pt"

    agent = CooAgent(
        agent_cfg,
        skip_base_strategy_model_cache=skip_cache,
        **agent_kwargs,
    )
    LOG.info("[load_cicero] CooAgent 初始化完成")
    return agent
# if __name__ == "__main__":
#     """独立运行示例"""
#     import argparse

#     # logging.basicConfig(level=logging.INFO)
#     # ##
#     logging.basicConfig(level=logging.WARNING)


#     # 确保在项目根目录运行
#     project_root = "/workspace/Diplomacy/diplomacy_cicero"
#     current_dir = os.getcwd()

#     if not current_dir.endswith("diplomacy_cicero"):
#         if os.path.exists(project_root):
#             os.chdir(project_root)
#             print(f"[路径修复] 工作目录从 {current_dir} 切换到 {os.getcwd()}")

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--cfg",
#         type=str,
#         default="conf/common/agents/coo_agent.prototxt",
#         help="Cicero配置文件路径",
#     )
#     parser.add_argument(
#         "--power",
#         type=str,
#         default="FRANCE",
#         help="势力名称",
#     )
#     args = parser.parse_args()

#     # 1) 用统一入口加载 Cicero + blueprint
#     print(f"\n[1/2] 加载 Cicero Agent: {args.cfg}")
#     try:
#         agent = load_cicero(
#             cfg_path=args.cfg,
#             skip_cache=False,
#         )
#         print("  ✓ Agent 初始化成功")
#     except Exception as e:
#         print(f"  ✗ Agent 初始化失败: {e}")
#         import traceback

#         traceback.print_exc()
#         raise SystemExit(1)

#     # 2) 起一盘游戏，查看指定势力在当前局面下的动作价值
#     print(f"\n[2/2] 评估动作价值...")
#     game = pydipcc.Game()
#     print("current_phase =", game.get_current_phase())
#     dump_power_legal_support_orders(game, args.power)
#     dump_current_assets(game)



#     try:
#         action_values = agent.evaluate_action_values(
#             game,
#             agent_power=args.power,
#         )
#         print(f"\n✓ 完成！共找到 {len(action_values)} 个动作\n")
#     except Exception as e:
#         print(f"\n✗ 评估失败: {e}")
#         import traceback

#         traceback.print_exc()
#         raise SystemExit(1)
if __name__ == "__main__":
    import argparse
    import random

    logging.basicConfig(level=logging.WARNING)

    project_root = "/workspace/Diplomacy/diplomacy_cicero"
    current_dir = os.getcwd()
    if not current_dir.endswith("diplomacy_cicero"):
        if os.path.exists(project_root):
            os.chdir(project_root)
            print(f"[路径修复] 工作目录从 {current_dir} 切换到 {os.getcwd()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="conf/common/agents/coo_agent.prototxt")
    parser.add_argument("--power", type=str, default="FRANCE")
    parser.add_argument("--steps", type=int, default=50, help="处理多少个 phase（你说的50轮）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", type=str, default="bp", choices=["bp", "search_br"])
    parser.add_argument("--mode", type=str, default="sample", choices=["sample", "top1"])
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--dump_every", type=int, default=1, help="每几轮打印一次 assets/history")
    parser.add_argument("--others", type=str, default="random", choices=["random", "hold"], help="其他势力怎么下单")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n[1/2] 加载 Cicero Agent: {args.cfg}")
    agent = load_cicero(cfg_path=args.cfg, skip_cache=False)
    print("  ✓ Agent 初始化成功")

    print(f"\n[2/2] 开始跑 {args.steps} 个 phase，控制势力={args.power}")
    game = pydipcc.Game()

    # 给被控制势力一个持续的 agent_state（不要每轮重建）
    agent_state = agent.initialize_state(args.power)

    for t in range(args.steps):
        state = game.get_state()
        if state["name"] == "COMPLETED":
            break
        phase = game.get_current_phase()
        print("\n" + "=" * 100)
        print(f"[STEP {t+1:02d}/{args.steps}] phase={phase}")
        print("=" * 100)

        if args.dump_every > 0 and (t % args.dump_every == 0):
            dump_current_assets(game)
            dump_power_legal_support_orders(game, args.power)

        # 1) 我方势力选动作（用多动作候选 + 采样/Top1）
        try:
            orders, action = agent.choose_action(
                game,
                args.power,
                agent_state=agent_state,
                source=args.source,
                top_k=args.topk,
                mode=args.mode,
                verbose=True,     # 这一轮输出候选统计（含support占比）
            )
        except Exception as e:
            print(f"[WARN] choose_action failed: {e}. fallback to empty orders.")
            orders, action = [], None

        if orders:
            print(f"[CHOSEN] {args.power}: {orders}")
            game.set_orders(args.power, orders)

        # 2) 其他势力随便走（为了让局面推进）
        for p in POWERS:
            if p == args.power:
                continue
            if args.others == "hold":
                # 不下单，dipcc 通常会默认 hold/waive
                game.set_orders(p, [])
            else:
                other_orders = pick_random_legal_orders(game, p)
                game.set_orders(p, other_orders)

        # 3) 推进到下一 phase
        game.process()

        if args.dump_every > 0 and (t % args.dump_every == 0):
            # 打印最近 1-2 个 phase 的历史（你也可以调 last_n）
            dump_order_history(game, last_n=2)

    print("\n✅ 完成。最终 phase =", game.get_current_phase())
    dump_order_history(game, last_n=10)
