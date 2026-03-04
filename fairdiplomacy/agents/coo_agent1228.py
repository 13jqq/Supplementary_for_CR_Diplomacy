# coding=utf-8
from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import heyhi
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent
from fairdiplomacy.agents.coo_coalition import CoalitionSolver, CoalitionSolverConfig, DELTA
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

def _pick_orders_from_dist(dist: Dict[Any, float], *, top_k: int = 30, mode: str = "sample") -> List[str]:
    """从一个 action->prob 的分布里选一个 action，并返回 orders(list[str])。"""
    if not dist:
        return []
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    if top_k and top_k > 0:
        items = items[:top_k]

    if mode == "top1":
        action = items[0][0]
    else:
        weights = [max(0.0, p) for _, p in items]
        s = sum(weights)
        if s <= 0:
            action = random.choice([a for a, _ in items])
        else:
            r = random.random() * s
            cum = 0.0
            action = items[-1][0]
            for (a, _), w in zip(items, weights):
                cum += w
                if cum >= r:
                    action = a
                    break

    return list(action) if isinstance(action, (list, tuple)) else [action]

# ---------------------------
# Minimal CooAgent
# ---------------------------
class CooAgent(BQRE1PAgent):
    """
    Minimal COO Agent:
    - get plausible bp policy
    - (optional) run Cicero BR search to get agent_policy
    - sample/top1 from top-k actions
    """

    def choose_orders(
        self,
        game: "pydipcc.Game",
        power: str,
        agent_state: Any,
        *,
        source: str = "search_br",  # "search_br" or "bp"
        top_k: int = 30,
        mode: str = "sample",       # "sample" or "top1"
    ) -> Tuple[List[str], List[Tuple[Any, float]], str]:
        # 1) blueprint plausible joint-action distribution
        bp_policy: Dict[str, Dict[Any, float]] = self.get_plausible_orders_policy(
            game=game,
            agent_power=power,
            agent_state=agent_state,
        )
        dist: Dict[Any, float] = bp_policy.get(power, {}) or {}
        used_source = "bp"

        # 2) Cicero BR-search distribution (value model + search), fallback to bp on failure
        if source == "search_br":
            try:
                search_res = self.run_best_response_against_correlated_bilateral_search(
                    game=game,
                    agent_power=power,
                    bp_policy=bp_policy,
                    agent_state=agent_state,
                )
                agent_pols = search_res.get_agent_policy()
                if power in agent_pols and agent_pols[power]:
                    dist = agent_pols[power]
                    used_source = "search_br"
            except Exception:
                used_source = "bp"

        if not dist:
            return [], [], used_source

        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None and top_k > 0:
            items = items[:top_k]

        # choose action
        if mode == "top1":
            action = items[0][0]
        else:
            # sample from (non-negative) probs; normalize on the fly
            weights = [max(0.0, p) for _, p in items]
            s = sum(weights)
            if s <= 0:
                action = random.choice([a for a, _ in items])
            else:
                r = random.random() * s
                cum = 0.0
                action = items[-1][0]
                for (a, _), w in zip(items, weights):
                    cum += w
                    if cum >= r:
                        action = a
                        break

        orders = list(action) if isinstance(action, (list, tuple)) else [action]
        return orders, items, used_source


def load_cicero(cfg_path: str, *, skip_cache: bool = False) -> CooAgent:
    """从 coo_agent.prototxt 读取配置并构造 CooAgent(BQRE1PAgent)。"""
    """
    Load coo_agent.prototxt and build CooAgent (BQRE1PAgent-based).
    Supports config layouts:
      - full_cfg.agent.coo_agent
      - full_cfg.coo_agent
    """
    full_cfg = heyhi.load_config(cfg_path)
    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "coo_agent"):
        agent_cfg = full_cfg.agent.coo_agent
    elif hasattr(full_cfg, "coo_agent"):
        agent_cfg = full_cfg.coo_agent
    else:
        raise ValueError(f"Bad config structure in {cfg_path}")

    return CooAgent(agent_cfg, skip_base_strategy_model_cache=skip_cache)





def _format_action_dist(items: List[Tuple[Any, float]]) -> str:
    # items are already top-k sorted
    lines = []
    for a, p in items:
        # action a is typically a tuple of orders
        if isinstance(a, (list, tuple)):
            act_str = "[" + ", ".join(map(str, a)) + "]"
        else:
            act_str = str(a)
        lines.append(f"    p={p:.8f}  action={act_str}")
    return "\n".join(lines) if lines else "    <empty>"


def _format_state_brief(st: Dict[str, Any]) -> str:
    units = st.get("units", {}) or {}
    centers = st.get("centers", {}) or {}
    lines = []
    for p in POWERS:
        us = units.get(p, []) or []
        cs = centers.get(p, []) or []
        lines.append(f"  {p}: units({len(us)})={us} | SC({len(cs)})={cs}")
    return "\n".join(lines)




import re
from collections import defaultdict

PHASE_SEASON = {"S": "SPRING", "F": "FALL", "W": "WINTER"}
PHASE_KIND = {"M": "MOVEMENT", "R": "RETREAT", "A": "ADJUSTMENT"}

def _parse_phase_meta(phase: str) -> Dict[str, Any]:
    """把 S1901M/F1902R/W1903A 解析为 {season, year, kind, label}（用于日志 + 判断 MOVEMENT）。"""
    """
    Supports:
      - Short dipcc style: S1901M / F1902R / W1903A
      - Fallback: just returns raw
    """
    s = str(phase)
    m = re.match(r"^([SFW])(\d{4})([MRA])$", s)
    if m:
        season_letter, year, kind_letter = m.group(1), int(m.group(2)), m.group(3)
        season = PHASE_SEASON.get(season_letter, season_letter)
        kind = PHASE_KIND.get(kind_letter, kind_letter)
        label = f"{season} {year} {kind}"
        return {"raw": s, "season": season, "year": year, "kind": kind, "label": label}

    # Sometimes dipcc prints long phase strings; do a best-effort parse
    up = s.upper()
    kind = "UNKNOWN"
    if "MOVEMENT" in up: kind = "MOVEMENT"
    elif "RETREAT" in up: kind = "RETREAT"
    elif "ADJUST" in up: kind = "ADJUSTMENT"
    season = "UNKNOWN"
    if "SPRING" in up: season = "SPRING"
    elif "FALL" in up or "AUTUMN" in up: season = "FALL"
    elif "WINTER" in up: season = "WINTER"
    year = None
    ym = re.search(r"(18|19|20)\d{2}", up)
    if ym:
        year = int(ym.group(0))
    label = f"{season} {year if year is not None else ''} {kind}".strip()
    return {"raw": s, "season": season, "year": year, "kind": kind, "label": label}


def _is_game_over(game: "pydipcc.Game") -> bool:
    """尽可能用 dipcc API 判断 game 是否结束。"""
    # Try common dipcc APIs; fall back to phase string checks
    for attr in ("is_game_done", "is_game_over", "game_over"):
        if hasattr(game, attr):
            try:
                v = getattr(game, attr)
                return bool(v() if callable(v) else v)
            except Exception:
                pass
    ph = str(game.get_current_phase())
    return ph.upper() in {"COMPLETED", "DONE", "END"} or "COMPLETED" in ph.upper()


def _normalize_unit_str(u: str) -> str:
    # dislodged units are marked like "*A MUN"
    return u[1:].strip() if u.startswith("*") else u.strip()


def _split_units(units: List[str]) -> Tuple[List[str], List[str]]:
    normal, dislodged = [], []
    for u in units or []:
        if isinstance(u, str) and u.startswith("*"):
            dislodged.append(_normalize_unit_str(u))
        else:
            normal.append(_normalize_unit_str(u) if isinstance(u, str) else str(u))
    return normal, dislodged


def _sc_owner_map(centers: Dict[str, List[str]]) -> Dict[str, str]:
    owner = {}
    if not centers:
        return owner
    for p, cs in centers.items():
        for c in cs or []:
            owner[str(c)] = str(p)
    return owner




def _diff_owner_map(prev: Dict[str, str] | None, cur: Dict[str, str]) -> str:
    if not prev:
        return "  (no previous SC snapshot)"
    gained = defaultdict(list)
    lost = defaultdict(list)
    for sc, cur_p in cur.items():
        prev_p = prev.get(sc, None)
        if prev_p is None:
            gained[cur_p].append(sc)
        elif prev_p != cur_p:
            lost[prev_p].append(sc)
            gained[cur_p].append(sc)
    if not gained and not lost:
        return "  (no SC ownership changes)"
    lines = []
    if gained:
        lines.append("  GAINED:")
        for p in sorted(gained.keys()):
            lines.append(f"    {p}: {sorted(gained[p])}")
    if lost:
        lines.append("  LOST:")
        for p in sorted(lost.keys()):
            lines.append(f"    {p}: {sorted(lost[p])}")
    return "\n".join(lines)


def _build_counts(st: Dict[str, Any]) -> Dict[str, int]:
    """
    builds = #centers - #units (ignore dislodged marker)
    Positive => builds, Negative => disbands
    """
    units = st.get("units", {}) or {}
    centers = st.get("centers", {}) or {}
    out = {}
    for p, cs in centers.items():
        us = units.get(p, []) or []
        normal, dislodged = _split_units(us)
        out[str(p)] = len(cs or []) - len(normal)  # dislodged should not count as stable units
    return out


def _get_orderable_locs(game: "pydipcc.Game", power: str) -> List[str]:
    for attr in ("get_orderable_locations", "get_orderable_locs"):
        if hasattr(game, attr):
            try:
                v = getattr(game, attr)
                res = v(power) if callable(v) else v
                return list(res) if res is not None else []
            except Exception:
                pass
    return []


def _get_possible_orders(game: "pydipcc.Game", power: str) -> List[str]:
    """
    Best-effort across possible dipcc bindings.
    For RETREAT/ADJUSTMENT this list is usually small and very useful to log.
    """
    # 1) get_all_possible_orders()
    if hasattr(game, "get_all_possible_orders"):
        try:
            allp = game.get_all_possible_orders()
            if isinstance(allp, dict) and power in allp:
                v = allp[power]
                if isinstance(v, dict):
                    # by location -> list
                    out = []
                    for lst in v.values():
                        out.extend(list(lst or []))
                    return out
                return list(v or [])
        except Exception:
            pass

    # 2) get_possible_orders(power)
    if hasattr(game, "get_possible_orders"):
        try:
            v = game.get_possible_orders(power)
            if isinstance(v, dict):
                out = []
                for lst in v.values():
                    out.extend(list(lst or []))
                return out
            return list(v or [])
        except Exception:
            pass

    return []


@dataclass
class Order:
    raw: str                      # 原始字符串
    unit_type: str                # "A" / "F"
    src: str                      # 源位置，例如 "PAR" / "STP/SC"
    action: str                   # 只允许: "H","-","S","C","R","D","B",""
    dst: Optional[str] = None     # 目的地（- / S(支援移动) / C / R）
    via: bool = False             # 仅用于 "-"（MOVE VIA）

    # 对 S / C：被支援/被运输的目标单位
    aux_unit_type: Optional[str] = None  # "A"/"F"
    aux_src: Optional[str] = None        # 目标单位所在位置

    # 对 S：支援的是 hold 还是 move（用单字符规范）
    #   支援hold => aux_action="H"
    #   支援move => aux_action="-"
    aux_action: Optional[str] = None



def parse_order(order: str, *, strict: bool = True) -> Order:
    """
    解析 dipcc 的单条 order 字符串。
    统一 action 为单字符：
      H  : hold
      -  : move（via=True 表示 VIA）
      S  : support（dst!=None 表示支援move；dst==None 表示支援hold；末尾H可有可无）
      C  : convoy
      R  : retreat
      D  : disband
      B  : build
      "" : empty

    strict=True：遇到未知模板直接 RuntimeError（用于“见到新模板就停程序”）。
    """
    s = (order or "").strip()
    if not s:
        return Order(raw=s, unit_type="", src="", action="")

    toks = s.split()
    if len(toks) < 2:
        if strict:
            raise RuntimeError(f"[ORDER-PARSE-ERROR] bad order: `{s}`")
        return Order(raw=s, unit_type="", src="", action="")

    ut, src = toks[0], toks[1]
    if ut not in ("A", "F"):
        if strict:
            raise RuntimeError(f"[ORDER-PARSE-ERROR] unknown unit_type: `{s}`")
        return Order(raw=s, unit_type=ut, src=src, action="")

    # Hold: A VEN H
    if len(toks) == 3 and toks[2] == "H":
        return Order(raw=s, unit_type=ut, src=src, action="H")

    # Move: F BRE - ENG  / Move Via: A APU - TUN VIA
    if len(toks) >= 4 and toks[2] == "-":
        dst = toks[3]
        via = (len(toks) >= 5 and toks[4] == "VIA")
        if strict and len(toks) not in (4, 5):
            raise RuntimeError(f"[ORDER-PARSE-ERROR] unknown MOVE form: `{s}`")
        return Order(raw=s, unit_type=ut, src=src, action="-", dst=dst, via=via)

    # Support:
    #   Support Hold: A RUH S A BEL      或 A RUH S A BEL H
    #   Support Move: F SEV S A UKR - RUM
    if len(toks) >= 5 and toks[2] == "S":
        aux_ut, aux_src = toks[3], toks[4]

        # S + 支援hold：len=5 或 len=6(末尾H)
        if len(toks) in (5, 6):
            if len(toks) == 6 and toks[5] != "H":
                if strict:
                    raise RuntimeError(f"[ORDER-PARSE-ERROR] unknown S_HOLD form: `{s}`")
            return Order(
                raw=s, unit_type=ut, src=src, action="S",
                aux_unit_type=aux_ut, aux_src=aux_src
            )

        # S + 支援move："... S A UKR - RUM"
        if len(toks) == 7 and toks[5] == "-":
            return Order(
                raw=s, unit_type=ut, src=src, action="S",
                aux_unit_type=aux_ut, aux_src=aux_src, dst=toks[6]
            )

        if strict:
            raise RuntimeError(f"[ORDER-PARSE-ERROR] unknown SUPPORT form: `{s}`")
        return Order(raw=s, unit_type=ut, src=src, action="S", aux_unit_type=aux_ut, aux_src=aux_src)

    # Convoy: F ION C A APU - TUN
    if len(toks) == 7 and toks[2] == "C" and toks[5] == "-":
        aux_ut, aux_src, dst = toks[3], toks[4], toks[6]
        return Order(
            raw=s, unit_type=ut, src=src, action="C",
            aux_unit_type=aux_ut, aux_src=aux_src, dst=dst
        )

    # Retreat: A MUN R SIL
    if len(toks) == 4 and toks[2] == "R":
        return Order(raw=s, unit_type=ut, src=src, action="R", dst=toks[3])

    # Disband: F BLA D
    if len(toks) == 3 and toks[2] == "D":
        return Order(raw=s, unit_type=ut, src=src, action="D")

    # Build: A WAR B
    if len(toks) == 3 and toks[2] == "B":
        return Order(raw=s, unit_type=ut, src=src, action="B")

    if strict:
        raise RuntimeError(f"[ORDER-PARSE-ERROR] Unknown order template: `{s}`")
    return Order(raw=s, unit_type=ut, src=src, action="")


def orders_to_tags(orders: List[str], *, strict: bool = True) -> List[str]:
    """
    把 orders(list[str]) 转成语义标签 list[str]（与 orders 等长）。

    标签集合（固定）：
      hold, move, move_via,
      support_hold, support_move,
      convoy, retreat, disband, build,
      empty
    """
    out: List[str] = []
    for s in orders or []:
        o = parse_order(s, strict=strict)

        if o.action == "H":
            out.append("hold")
        elif o.action == "-":
            out.append("move_via" if getattr(o, "via", False) else "move")
        elif o.action == "S":
            out.append("support_move" if getattr(o, "dst", None) else "support_hold")
        elif o.action == "C":
            out.append("convoy")
        elif o.action == "R":
            out.append("retreat")
        elif o.action == "D":
            out.append("disband")
        elif o.action == "B":
            out.append("build")
        elif o.action == "":
            out.append("empty")
        else:
            # strict=True 基本到不了这；留兜底
            out.append(f"unknown({o.action})")

    return out



from typing import Any, Dict, List, Tuple

def get_unitlocation_SClocation(state_before: Dict[str, Any]) -> Tuple[
    Dict[str, str],                      # unit_loc_owner: loc -> power
    Dict[str, str],                      # sc_owner: sc -> power
    Dict[str, Dict[str, List[str]]],     # power_assets: power -> {"unit_locs":[...], "scs":[...]}
]:
    """
    一次性提取（最精简）：
    1) unit_loc_owner: loc -> power     （谁的单位站在这个格子上）
    2) sc_owner:       sc  -> power     （这个补给中心归谁）
    3) power_assets:   power -> {"unit_locs": [...], "scs": [...]}  （按国家聚合，便于写log/后续检测）

    注意：
    - unit_locs 是“单位占位”，不是领土归属；
    - scs 是“SC归属”，用于判断是否有人 move 到你的 SC；
    - dislodged 单位可能形如 "*A MUN"，这里会去掉 "*"。
    """
    units_by_power = state_before.get("units") or {}
    centers_by_power = state_before.get("centers") or {}

    unit_loc_owner: Dict[str, str] = {}
    sc_owner: Dict[str, str] = {}
    power_assets: Dict[str, Dict[str, List[str]]] = {p: {"unit_locs": [], "scs": []} for p in POWERS}

    # 1) 单位占位：loc -> power 以及按 power 聚合 unit_locs
    for power, units in units_by_power.items():
        p = str(power)
        power_assets.setdefault(p, {"unit_locs": [], "scs": []})
        for u in units or []:
            if not isinstance(u, str):
                continue
            s = u.strip()
            if s.startswith("*"):
                s = s[1:].strip()  # "*A MUN" -> "A MUN"
            toks = s.split()
            if len(toks) >= 2:
                loc = toks[1]
                unit_loc_owner[loc] = p
                power_assets[p]["unit_locs"].append(loc)

    # 2) SC 归属：sc -> power 以及按 power 聚合 scs
    for power, scs in centers_by_power.items():
        p = str(power)
        power_assets.setdefault(p, {"unit_locs": [], "scs": []})
        for sc in scs or []:
            sc = str(sc)
            sc_owner[sc] = p
            power_assets[p]["scs"].append(sc)

    # 3) 排序一下便于 log 阅读（可选，但很推荐）
    for p in power_assets:
        power_assets[p]["unit_locs"] = sorted(set(power_assets[p]["unit_locs"]))
        power_assets[p]["scs"] = sorted(set(power_assets[p]["scs"]))

    return unit_loc_owner, sc_owner, power_assets

def territory_from_state(state: dict, me: str) -> set[str]:
    """
    极简版：我方领土 = centers[me] ∪ influence[me]（如果 influence 不存在就只用 centers）
    """
    centers = set((state.get("centers", {}) or {}).get(me, []) or [])
    influence_dict = state.get("influence", None)

    if isinstance(influence_dict, dict):
        influence = set(influence_dict.get(me, []) or [])
        return centers | influence

    # 没有 influence 字段就退化为 centers（你说先不折腾验证）
    return centers

def last_round_support_attack(
    state_before: Dict[str, Any],
    last_orders: Dict[str, List[str]],
    me: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    区分“有用支援/无用支援/攻击”：

    1) 有用支援（support_used）
       - 对手 S(支援) / C(运输) 的目标单位是我方单位（用我方上一回合 orders 来确认）
       - 且支援方向与我方该单位的指令一致：
         * 我方该单位 Hold  => 对手是 Support Hold
         * 我方该单位 Move A->B => 对手是 Support Move 到同一个 B / Convoy 到同一个 B

    2) 无用支援（support_unused）
       - 对手 S/C 的目标单位看起来在支援我方（目标格能对应到我方单位/领土），但方向不一致或无法确认

    3) 攻击（attackers）
       - 直接 move dst 落我方 territory
       - 或 support-move / convoy 的 dst 落我方 territory
    """
    my_terr = set(territory_from_state(state_before, me))

    # NEW: who occupies each location in state_before
    unit_loc_owner: Dict[str, str] = {}
    units_by_power = state_before.get("units") or {}
    for pp, units in units_by_power.items():
        for u in units or []:
            if not isinstance(u, str):
                continue
            uu = u.strip()
            if uu.startswith("*"):
                uu = uu[1:].strip()  # "*A BRE" -> "A BRE"
            toks = uu.split()
            if len(toks) >= 2:
                loc = toks[1]
                unit_loc_owner[loc] = str(pp)


    support_used, support_unused, attackers = defaultdict(list), defaultdict(list), defaultdict(list)

    # --- 关键：解析我方上一回合 orders，建立 “单位位置 -> 我方意图指令” 的映射 ---
    my_intent_by_src: Dict[str, Order] = {}
    for od in (last_orders or {}).get(me, []) or []:
        oo = parse_order(od, strict=False)
        if oo.src:
            # Diplomacy 同一格不会有两支单位，src 当 key 足够
            my_intent_by_src[oo.src] = oo

    def _is_aligned_support(support_order: Order, my_order: Order) -> bool:
        """判断对手的 S/C 是否与我方该单位指令一致。"""
        # 支援 Hold：对手 S A X（无 dst），我方该单位必须是 Hold
        if support_order.action == "S" and not support_order.dst:
            return my_order.action == "H"

        # 支援 Move：对手 S A X - Y，我方该单位必须 Move 到同一个 Y
        if support_order.action == "S" and support_order.dst:
            return (my_order.action == "-" and my_order.dst == support_order.dst)

        # Convoy：对手 C A X - Y，我方该单位必须 Move 到同一个 Y（不强制 VIA，防止格式差异）
        if support_order.action == "C" and support_order.dst:
            return (my_order.action == "-" and my_order.dst == support_order.dst)

        return False

    for p, ords in (last_orders or {}).items():
        if p == me:
            continue
        for s in ords or []:
            o = parse_order(s, strict=False)

            # -----------------------------
            # 1) 先处理 S/C：判断是否“对齐我方意图”
            # -----------------------------
            if o.action in ("S", "C"):
                # aux_src 是被支援/被运输的那支单位所在格
                if o.aux_src:
                    my_order = my_intent_by_src.get(o.aux_src, None)

                    # A) 如果 aux_src 对应我方上一回合某个单位的指令：可以做对齐判断
                    if my_order is not None:
                        if _is_aligned_support(o, my_order):
                            support_used[p].append(s)
                        else:
                            support_unused[p].append(s)
                        continue  # 不再往下当作 attack 处理（避免重复计数）

                    # B) 如果对不上我方单位，但 aux_src 在我方 territory：当作“可能支援但无法确认/方向不一致”
                    if o.aux_src in my_terr:
                        occ = unit_loc_owner.get(o.aux_src, None)

                        # 只有占位单位是我方，才可能算“支援我但没对齐/无法确认”
                        if occ == me:
                            support_unused[p].append(s)
                        else:
                            # 敌方单位(或未知)在我领土/SC上：这是在“加固敌人”，应该算攻击（让联盟分数扣分）
                            attackers[p].append(s)
                        continue


                # C) 如果 S/C 的 dst 落我方 territory：这是“帮敌进我领土”（攻击口径）
                if o.dst and (o.dst in my_terr):
                    attackers[p].append(s)
                    continue

            # -----------------------------
            # 2) 处理直接 move 进我方 territory：攻击
            # -----------------------------
            if o.action == "-" and o.dst and (o.dst in my_terr):
                attackers[p].append(s)

    return dict(support_used), dict(support_unused), dict(attackers)


# ---------------------------
# Main loop: run-to-end + log.txt (UPDATED)
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="conf/common/agents/coo_agent.prototxt")
    parser.add_argument("--project_root", type=str, default="/workspace/Diplomacy/diplomacy_cicero")

    # POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
    parser.add_argument("--power", type=str, default="AUSTRIA", choices=POWERS)

    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--source", type=str, default="search_br", choices=["search_br", "bp"])
    # parser.add_argument("--mode", type=str, default="sample", choices=["sample", "top1"])
    
    parser.add_argument("--topk", type=int, default=30)

    # parser.add_argument("--others", type=str, default="base", choices=["base", "hold"])

    # ---- policy config (me) ----
    parser.add_argument("--me_source", type=str, default="search_br",
                        choices=["search_br", "bp"],
                        help="my power policy source")
    parser.add_argument("--me_mode", type=str, default="sample",
                        choices=["sample", "top1"],
                        help="how to pick action from dist for my power")

    # ---- policy config (others) ----
    parser.add_argument("--other_source", type=str, default="search_br",
                        choices=["bp", "search_br"],
                        help="other powers policy source (search_br is VERY slow)")
    parser.add_argument("--other_mode", type=str, default="sample",
                        choices=["sample", "top1"],
                        help="how to pick action from dist for other powers")


    parser.add_argument("--max_phases", type=int, default=200)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log", type=str, default=None)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pause", action="store_true")  


    args = parser.parse_args()
    random.seed(args.seed)

    # --- always chdir first ---
    if args.project_root and os.path.exists(args.project_root):
        os.chdir(args.project_root)

    # --- always compute log_path (fixes end-of-run UnboundLocalError) ---
    ts = datetime.now().strftime("%y%m%d%H%M")
    log_dir = args.log_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(os.getcwd(), log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = args.log if args.log else os.path.join(log_dir, f"{ts}.log")
    print(f"[LOG] writing to: {log_path}")
    start_time = datetime.now()  # ✅ 记录开始时间

    self_agent = load_cicero(args.cfg, skip_cache=False)

    game = pydipcc.Game()
    # self_state = self_agent.initialize_state(args.power)
    states = {p: self_agent.initialize_state(p) for p in POWERS}
    self_state = states[args.power]

    prev_sc_owner = None  # for SC diff
    step = 0

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== RUN START ===\n")
        f.write(f"start_time={start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")  # ✅ 写入开始时间
        f.write(f"cwd={os.getcwd()}\n")
        f.write(f"cfg={args.cfg}\n")
        f.write(
            f"power={args.power}, seed={args.seed}, me_source={args.me_source}, me_mode={args.me_mode}, "
            f"other_source={args.other_source}, other_mode={args.other_mode}, "
            f"topk={args.topk}, max_phases={args.max_phases}\n\n"
        )

        f.flush()
        coalition_solver = CoalitionSolver(CoalitionSolverConfig(top_k=4, min_coalition_size=1, max_coalition_size=2))
        last_set_orders = {}  # optional: 给 H_t 占位用
        movement_step = 0
        phase_step = 0
        prev_state_before = None
        prev_set_orders = None
        prev_phase = None
        attack_hist = defaultdict(list)   # power -> ["S1901M: A PAR - BUR", ...]


        while True:
            # termination guards (pre)
            if step >= args.max_phases:
                f.write(f"\n[STOP] reached max_phases={args.max_phases}\n")
                break
            if _is_game_over(game):
                f.write("\n[STOP] game over detected\n")
                break

            st = game.get_state()
            phase = game.get_current_phase()
            meta = _parse_phase_meta(str(phase))
            is_movement = (meta.get("kind") == "MOVEMENT")

            # --- me territory (给联盟算法用) ---
            my_territory = territory_from_state(st, args.power)



            # --- last round support/attack（先给默认值，避免第一轮 UnboundLocalError）---
            support_used: Dict[str, List[str]] = {}
            support_unused: Dict[str, List[str]] = {}
            attack_last: Dict[str, List[str]] = {}

            # 备份“上一回合”的快照（注意：后面你会把 prev_* 更新成当前回合）
            last_phase_str = prev_phase
            last_orders_all = prev_set_orders
            last_state_before = prev_state_before

            if last_state_before is not None and last_orders_all is not None and last_phase_str is not None:
                support_used, support_unused, attack_last = last_round_support_attack(
                    last_state_before, last_orders_all, args.power
                )

                for p, ords in (attack_last or {}).items():
                    attack_hist[p] += [f"{last_phase_str}: {o}" for o in ords]
            #以上为验证last round support/attack功能#

            # --- policy ---
            bp_policy: Dict[str, Dict[Any, float]] = self_agent.get_plausible_orders_policy(
                game=game,
                agent_power=args.power,
                agent_state=states[args.power],
            )


            H_t = {
                "last_orders": last_set_orders,
                "territory": my_territory,
                "support_used": support_used,
                "support_unused": support_unused,
                "attack_last": attack_last,

                "attack_hist": dict(attack_hist),
            }

            g = coalition_solver.solve_coalition(
                game=game,
                me=args.power,
                pi_me=args.me_source,  # 占位
                pi_bp=bp_policy,       # 先给进去，后面 rollout 会用
                H_t=H_t,
                step=movement_step,
            )

            # 写入同一个 log 文件（候选盟友来自 E）
            cands = [j for (_, j) in g.E]
            cand_str = ", ".join([
                f"{j}(EU={g.w.get((args.power, j), g.w.get((j, args.power), 0.0)):.2f})"
                for j in cands
            ])



            used_source = "bp"
            my_dist = bp_policy.get(args.power, {}) or {}

            # search_br only for movement
            if args.debug:
                print(f"\n[DBG] phase={phase} label={meta.get('label')} kind={meta.get('kind')} is_movement={is_movement} args.source={args.me_source}")
                print(f"[DBG] will_try_search_br? {args.me_source=='search_br' and is_movement}")
            if args.pause:
                input("[DBG] enter to continue ... ")

            if args.me_source == "search_br" and is_movement:
                try:
                    if args.debug:
                        print("[DBG] running run_best_response_against_correlated_bilateral_search(...)")
                    search_res = self_agent.run_best_response_against_correlated_bilateral_search(
                        game=game,
                        agent_power=args.power,
                        bp_policy=bp_policy,
                        agent_state=states[args.power],
                    )
                    agent_pols = search_res.get_agent_policy()
                    if args.debug:
                        print(f"[DBG] search_res.get_agent_policy keys={list(agent_pols.keys())}")
                        print(f"[DBG] agent_pols[{args.power}] empty? {not bool(agent_pols.get(args.power))}")

                    if agent_pols.get(args.power):
                        my_dist = agent_pols[args.power]
                        used_source = "search_br"
                    else:
                        used_source = "bp"
                except Exception as e:
                    if args.debug:
                        print(f"[DBG] search_br failed -> fallback bp. err={repr(e)}")
                    used_source = "bp"
            elif args.me_source == "search_br" and (not is_movement):
                if args.debug:
                    print(f"[DBG] non-movement => skip search_br, use bp")
                used_source = "bp"
            if args.debug:
                print(f"[DBG] used_source={used_source}  dist_size={len(my_dist) if my_dist else 0}")
                if args.pause:
                    input("[DBG] enter to continue ... ")


            my_items = sorted(my_dist.items(), key=lambda kv: kv[1], reverse=True)
            if args.topk and args.topk > 0:
                my_items = my_items[:args.topk]
            my_orders = _pick_orders_from_dist(my_dist, top_k=args.topk, mode=args.me_mode)

            # --- set orders for all powers ---
            set_orders: Dict[str, List[str]] = {p: [] for p in POWERS}

            set_orders[args.power] = my_orders
            game.set_orders(args.power, my_orders)

            for p in POWERS:
                if p == args.power:
                    continue

                other_dist = bp_policy.get(p, {}) or {}

                # 可选：让其他国家也跑 search_br（会非常慢）
                if args.other_source == "search_br" and is_movement:
                    try:
                        bp_policy_p = self_agent.get_plausible_orders_policy(
                            game=game, agent_power=p, agent_state=states[p]
                        )
                        search_res_o = self_agent.run_best_response_against_correlated_bilateral_search(
                            game=game,
                            agent_power=p,
                            bp_policy=bp_policy_p,   # ✅ 用 p 自己视角的 bp_policy
                            agent_state=states[p],
                        )
                        other_pols = search_res_o.get_agent_policy()
                        if other_pols.get(p):
                            other_dist = other_pols[p]
                    except Exception:
                        pass


                o = _pick_orders_from_dist(other_dist, top_k=args.topk, mode=args.other_mode)
                set_orders[p] = o
                game.set_orders(p, o)

            # ✅ 只在“所有国家 orders 都 set 完”之后，记录本回合（给下一回合用）
            prev_state_before = st
            prev_set_orders = set_orders
            prev_phase = str(phase)


            # --- log BEFORE process ---
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"[PHASE {step:04d}] phase={phase} | {meta.get('label')}\n")
            f.write(f"[PHASE META] kind={meta.get('kind')} season={meta.get('season')} year={meta.get('year')}\n")
            # ---------------------------
            # DEBUG: 打印“上一回合”的支援/攻击判定结果
            # ---------------------------
            if last_phase_str is None:
                f.write("[LAST ROUND SUPPORT/ATTACK] prev_phase=None (game just started)\n")
            else:
                # 我方上一回合每个单位(src)的真实指令（用于对照：对手的 S/C 是否与我一致）
                my_prev_intent_by_src: Dict[str, str] = {}
                for od in (last_orders_all or {}).get(args.power, []) or []:
                    oo = parse_order(od, strict=False)
                    if oo.src:
                        my_prev_intent_by_src[oo.src] = od

                f.write(f"[LAST ROUND SUPPORT/ATTACK] prev_phase={last_phase_str}\n")
                # NEW: build quick lookup for (power, order) -> delta, and power -> T_new
                ev = (g.meta.get("evidence", {}) or {})
                ev_map = {
                    pp: {it["order"]: float(it.get("delta", 0.0)) for it in (ev.get(pp, []) or [])}
                    for pp in POWERS
                }
                T_new_map = (g.meta.get("T_new", {}) or {})


                for p in POWERS:
                    if p == args.power:
                        continue
                    used_list = (support_used or {}).get(p, []) or []
                    unused_list = (support_unused or {}).get(p, []) or []
                    atk_list = (attack_last or {}).get(p, []) or []

                    if not used_list and not unused_list and not atk_list:
                        continue

                    f.write(f"  {p}:\n")
                    for s in used_list:
                        so = parse_order(s, strict=False)
                        my_od = my_prev_intent_by_src.get(getattr(so, "aux_src", None), None)
                        d = ev_map.get(p, {}).get(s, 0.0)
                        tnew = T_new_map.get(p, None)
                        f.write(
                            f"    USED   (+) {s}"
                            + (f"  | my={my_od}" if my_od else "  | my=<NONE>")
                            + f"  | delta={d:+.1f}"
                            + (f"  | T_new={tnew:+.2f}\n" if tnew is not None else "\n")
                        )


                    for s in unused_list:
                        so = parse_order(s, strict=False)
                        my_od = my_prev_intent_by_src.get(getattr(so, "aux_src", None), None)
                        d = ev_map.get(p, {}).get(s, 0.0)
                        tnew = T_new_map.get(p, None)
                        f.write(
                            f"    UNUSED (+) {s}"
                            + (f"  | my={my_od}" if my_od else "  | my=<NONE>")
                            + f"  | delta={d:+.1f}"
                            + (f"  | T_new={tnew:+.2f}\n" if tnew is not None else "\n")
                        )


                    for s in atk_list:
                        d = ev_map.get(p, {}).get(s, 0.0)
                        tnew = T_new_map.get(p, None)
                        f.write(
                            f"    ATTACK (-) {s}"
                            + f"  | delta={d:+.1f}"
                            + (f"  | T_new={tnew:+.2f}\n" if tnew is not None else "\n")
                        )

            # Retreat / Adjustment extra diagnostics
            if meta.get("kind") in ("RETREAT", "ADJUSTMENT"):
                f.write("[PHASE REQUIREMENTS]\n")
                builds = _build_counts(st)
                for p in POWERS:
                    locs = _get_orderable_locs(game, p)
                    poss = _get_possible_orders(game, p)
                    if locs or poss or (meta.get("kind") == "ADJUSTMENT" and builds.get(p, 0) != 0):
                        f.write(f"  {p}: orderable_locs={locs}\n")
                        if meta.get("kind") == "ADJUSTMENT":
                            b = builds.get(p, 0)
                            if b != 0:
                                f.write(f"    build_count={b}  ({'BUILD' if b>0 else 'DISBAND'} x{abs(b)})\n")
                        if poss:
                            f.write(f"    possible_orders({len(poss)}): {poss}\n")

            # state
            f.write(f"[STATE BEFORE]\n{_format_state_brief(st)}\n")
            
            f.write(f"[TERR] phase={phase} me={args.power} terrN={len(my_territory)} sample={sorted(my_territory)[:8]}\n")
            f.write(
                f"[ALLIANCE] phase_step={phase_step} movement_step={movement_step} phase={g.meta['phase']} "
                f"candidates=[{cand_str}] chosen={g.chosen_allies}\n"
            )
            # --- alliance score evidence log (order -> delta) ---
            ev = (g.meta.get("evidence", {}) or {})
            T0 = (g.meta.get("T_prev", {}) or {})
            T1 = (g.meta.get("T_new", {}) or {})
            if ev:
                f.write("[ALLIANCE SCORE UPDATES]\n")
                for j in cands:
                    if j in T0 and j in T1:
                        f.write(f"  {j}: T {T0[j]:+.2f} -> {T1[j]:+.2f}\n")
                    for item in ev.get(j, []):
                        f.write(f"    {item['order']} -> {item['delta']:+.1f} ({item['tag']})\n")


            f.write(f"[MY ACTION DIST] used_source={used_source}, topk={len(my_items)}\n")
            f.write(_format_action_dist(my_items) + "\n")

            f.write("[ORDERS SET]\n")
            for p in POWERS:
                f.write(f"  {p}: {set_orders[p]}\n")
                # f.write(f" {p}: {orders_to_tags(set_orders[p], strict=True)}\n")

            # --- advance ---
            try:
                game.process()
            except Exception as e:
                f.write(f"[ERROR] game.process() failed @phase={phase} ({meta.get('label')}): {repr(e)}\n")
                break
            if is_movement:
                movement_step += 1
            phase_step += 1
            last_set_orders = set_orders


            # --- log AFTER process immediately (this is the key improvement for retreat/winter visibility) ---
            st_after = game.get_state()
            # --- dislodged/retreat diagnostics ---
            units_after = (st_after.get("units", {}) or {})
            dislodged_by_power = {}
            for p in POWERS:
                _, dislodged = _split_units(units_after.get(p, []) or [])
                if dislodged:
                    dislodged_by_power[p] = dislodged

            if dislodged_by_power:
                f.write("[DISLODGED / RETREAT INFO]\n")
                for p, ds in dislodged_by_power.items():
                    poss = _get_possible_orders(game, p)  # in retreat phase, this will list retreat/disband options
                    for u in ds:
                        # u like "A MUN" / "F ENG" etc
                        toks = u.split()
                        loc = toks[1] if len(toks) >= 2 else "<?>"
                        # retreat/disband orders usually start with "A MUN" or "F XXX"
                        opts = [o for o in poss if isinstance(o, str) and o.startswith(u + " ")]
                        f.write(f"  {p}: dislodged={u}  options={opts}\n")




            phase_after = game.get_current_phase()
            meta_after = _parse_phase_meta(str(phase_after))
            sc_owner_after = _sc_owner_map(st_after.get("centers", {}) or {})

            f.write(f"[STATE AFTER]\n{_format_state_brief(st_after)}\n")
            f.write("[SC CHANGES (prev -> after)]\n")
            f.write(_diff_owner_map(prev_sc_owner, sc_owner_after) + "\n")

            prev_sc_owner = sc_owner_after
            f.flush()

            step += 1

        end_time = datetime.now()  # ✅ 记录结束时间
        elapsed = end_time - start_time  # ✅ 计算耗时

        f.write(f"\nfinal_phase={game.get_current_phase()}\n")
        f.write(f"end_time={end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")  # ✅ 写入结束时间
        f.write(f"elapsed_time={elapsed}\n")  # ✅ 写入总耗时
        f.write("=== RUN END ===\n")
        f.flush()

if __name__ == "__main__":
    main()
