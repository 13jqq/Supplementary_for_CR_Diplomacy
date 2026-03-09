# coding=utf-8
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple
import heyhi
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent
from fairdiplomacy.utils.sampling import sample_p_dict

POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


class ConsistentAgent(BQRE1PAgent):
    """
    一致性校验智能体（骨架版）

    目标效果：在 Cicero/BQRE 给出的 top-k action 中，不直接用 top1，
            而是保留候选列表，后续我们将从 top 往下找“动作一致性更强”的 action。

    输出：choose_orders 返回 (orders, items, used_source)
         - orders: 最终选中的 orders(list[str])
         - items : top-k 候选 action 列表 [(action, prob), ...]（用于后续一致性筛选）
         - used_source: "bp" / "search_br" / "bqre_topK"
    """

    def choose_orders(
        self,
        game: "Any",          # pydipcc.Game
        power: str,
        agent_state: Any,
        *,
        source: str = "bqre_topK",   # ✅ 新增：默认走 bqre_topK（多 action 分布）
        top_k: int = 30,
        mode: str = "top1",          # "top1" 或 "sample"
    ) -> Tuple[List[str], List[Tuple[Any, float]], str, List[Tuple[Any, float, str]]]:
        # 1) 先拿 blueprint plausible joint-action 分布（Cicero 原流程）
        bp_policy: Dict[str, Dict[Any, float]] = self.get_plausible_orders_policy(
            game=game,
            agent_power=power,
            agent_state=agent_state,
        )
        dist: Dict[Any, float] = bp_policy.get(power, {}) or {}
        used_source = "bp"

        # 2) 三种来源可选：bqre_topK / search_br / bp
        # 2.1) ✅ 新增：bqre_topK —— 直接跑 BQRE 的 run_search 拿多 action 分布（推荐）
        if source == "bqre_topK":
            try:
                # 复用已算出的 bp_policy，避免重复计算 plausible orders
                res = self.run_search(
                    game,
                    bp_policy=bp_policy,
                    agent_power=power,
                    agent_state=agent_state,
                )
                dist = res.get_agent_policy().get(power, {}) or {}
                used_source = "bqre_topK"
            except Exception:
                # 失败则回退 bp
                dist = bp_policy.get(power, {}) or {}
                used_source = "bp"

        # 2.2) 保留原逻辑：search_br —— correlated BR（通常只给 best action 分布，可能退化）
        elif source == "search_br":
            try:
                search_res = self.run_best_response_against_correlated_bilateral_search(
                    game=game,
                    agent_power=power,
                    bp_policy=bp_policy,
                    agent_state=agent_state,
                )
                agent_pols = search_res.get_agent_policy()
                if agent_pols.get(power):
                    dist = agent_pols[power]
                    used_source = "search_br"
            except Exception:
                used_source = "bp"

        # 2.3) source == "bp"：保持 dist 为 bp

        if not dist:
            return [], [], used_source, []

        # 3) 截取 top-k 候选列表（关键：保留 items，后面我们就在这里做一致性校验）
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None and top_k > 0:
            items = items[:top_k]

        kept, dropped = filter_action_set_by_consistency(game, power, items)
        items = kept if kept else items   # kept 为空则回退


        # TODO（下一步实现）：从 items[0], items[1], ... 往下找“一致性通过校验”的 action
        # 目前骨架版：暂时仍然选 top1 / 或按概率 sample（方便你先跑通流程 + 打印 items）


        pool = items  # items 已经是 kept(优先) 或 fallback 后的列表

        # ✅ 用这段替换你原来的 if mode == "top1" ... else ...
        if mode in ("sample", "bqre"):
            dist2 = _renorm(pool)
            if dist2 is None:
                action = random.choice([a for a, _ in pool])
            else:
                action = sample_p_dict(dist2)   # BQRE-style sampling
        else:
            action = max(pool, key=lambda kv: kv[1])[0]  # top1

        # if mode == "top1":
        #     action = items[0][0]
        # else:
        #     weights = [max(0.0, p) for _, p in items]
        #     s = sum(weights)
        #     if s <= 0:
        #         action = random.choice([a for a, _ in items])
        #     else:
        #         r = random.random() * s
        #         cum = 0.0
        #         action = items[-1][0]
        #         for (a, _), w in zip(items, weights):
        #             cum += w
        #             if cum >= r:
        #                 action = a
        #                 break

        orders = list(action) if isinstance(action, (list, tuple)) else [action]
        return orders, items, used_source, dropped

def _renorm_action_items(items):
    d = {a: max(0.0, float(p)) for a, p in items}
    s = sum(d.values())
    if s <= 0:
        return None
    return {a: p / s for a, p in d.items()}
def load_cicero(cfg_path: str, *, skip_cache: bool = False) -> ConsistentAgent:
    """
    从 consistent_agent.prototxt 读取配置并构造 ConsistentAgent（BQRE1PAgent-based）
    目标效果：复用你现有“配置文件挂载 + agent 映射”那套流程，只是换成 consistent_agent。
    输出：ConsistentAgent 实例
    """
    full_cfg = heyhi.load_config(cfg_path)

    # 兼容两种常见配置结构：
    # 1) full_cfg.agent.consistent_agent
    # 2) full_cfg.consistent_agent
    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "consistent_agent"):
        agent_cfg = full_cfg.agent.consistent_agent
    elif hasattr(full_cfg, "consistent_agent"):
        agent_cfg = full_cfg.consistent_agent
    else:
        raise ValueError(f"Bad config structure in {cfg_path}: cannot find consistent_agent")

    return ConsistentAgent(agent_cfg, skip_base_strategy_model_cache=skip_cache)
# ====== 新增函数 1：位置规范化（只去掉 *，保留 /SC /NC /EC 等后缀）======
def _norm_loc(loc: str | None) -> str | None:
    if not loc:
        return None
    s = str(loc).strip()
    if s.startswith("*"):  # dipcc 有时会用 * 标记特殊状态
        s = s[1:].strip()
    return s  # ✅ 不再 split('/')，保留 STP/SC 这种形式


def _build_loc2power(game: Any) -> Dict[str, str]:
    """
    loc -> power（单值映射）：只表示“当前 unit 占位者”
    - ✅ 只用 units，不混 centers / influence（避免把“SC 所属”当成“被攻击对象”）
    - ✅ 保留 /SC /NC /EC 等后缀
    """
    try:
        st = game.get_state()
    except Exception:
        return {}

    if not isinstance(st, dict) and hasattr(st, "to_dict"):
        try:
            st = st.to_dict()
        except Exception:
            return {}
    if not isinstance(st, dict):
        return {}

    units = st.get("units")
    if not isinstance(units, dict):
        return {}

    loc2p: Dict[str, str] = {}
    for pwr, unit_list in units.items():
        for u in unit_list or []:
            parts = str(u).split()
            if len(parts) >= 2:
                loc = _norm_loc(parts[1])
                if loc:
                    loc2p[loc] = pwr
    return loc2p

def _owner_by_territory(prov_base: str | None, st: Dict[str, Any], occ: Dict[str, str], my_power: str) -> str | None:
    """
    统一的“目标地归属/控制者”判定（给 C1/C2 用的通用函数）
    优先级：
      1) 若该省当前有单位占位：按占位单位的国家算（最强语义：你在打谁的单位）
      2) 否则若是 SC：按 centers 的 owner 算
      3) 否则按 influence 的唯一 owner 算（若多国同时出现则视为不确定 -> None）
    返回：
      - 若判定为我方自己，则返回 None（因为 C1/C2 只关心“敌人是谁”）
    """
    if not prov_base:
        return None

    # 1) 看该省当前是否有单位占位（支持 STP vs STP/SC 这种 coast 形式）
    hit = None
    for k, p in occ.items():
        if k == prov_base or k.startswith(prov_base + "/"):
            if hit is None:
                hit = p
            elif hit != p:
                hit = None
                break
    if hit:
        return None if hit == my_power else hit

    # 2) SC owner（centers）
    centers = st.get("centers", {}) or {}
    if isinstance(centers, dict):
        for pwr, clist in centers.items():
            for c in (clist or []):
                if str(c).split("/")[0] == prov_base:
                    pwr = str(pwr)
                    return None if pwr == my_power else pwr

    # 3) nonSC control trace（influence）— 要求唯一
    influence = st.get("influence", None)
    if isinstance(influence, dict):
        owner = None
        for pwr, tlist in influence.items():
            for t in (tlist or []):
                if str(t).split("/")[0] == prov_base:
                    if owner is None:
                        owner = str(pwr)
                    elif owner != str(pwr):
                        return None  # 多国同时声称/出现 -> 不确定
        if owner:
            return None if owner == my_power else owner

    return None


def get_territory_parts(st: Dict[str, Any], power: str) -> Tuple[set[str], set[str], set[str]]:
    """
    返回三类集合（都用 base province，不保留 /SC /NC /EC 后缀）：
      - sc_set: 该国 SC 省份集合
      - unit_set: 该国当前 unit 所在省份集合
      - past_free_set: 该国过去占过/控制过、且目前没有任何玩家 unit 占位的省份集合
        * 依赖 st["influence"]（若不存在则返回空集合）
        * 另外会排除 sc_set 和 unit_set，保证三类尽量互斥（更干净）
    """
    units = st.get("units", {}) or {}
    centers = st.get("centers", {}) or {}
    influence = st.get("influence", None)

    # 1) SC
    sc_set = {str(x).split("/")[0] for x in (centers.get(power) or [])}

    # 2) 当前 unit 所在省份
    unit_set: set[str] = set()
    for u in (units.get(power) or []):
        s = str(u).strip().lstrip("*")
        parts = s.split()
        if len(parts) >= 2:
            unit_set.add(parts[1].split("/")[0])

    # 3) past_free：需要 influence 提供“历史/控制痕迹”
    if not isinstance(influence, dict):
        past_free_set = set()
        return sc_set, unit_set, past_free_set

    # 当前所有玩家 unit 占位（用 base province）
    occupied_now: set[str] = set()
    for pwr, ulist in units.items():
        for u in (ulist or []):
            s = str(u).strip().lstrip("*")
            parts = s.split()
            if len(parts) >= 2:
                occupied_now.add(parts[1].split("/")[0])

    # influence 视为“过去占过/控制过”的候选集合
    owned_hist = {str(x).split("/")[0] for x in (influence.get(power) or [])}

    # 过去占过且当前无人占位；同时排除 sc 和当前 unit（让三类更干净）
    past_free_set = (owned_hist - occupied_now) - sc_set - unit_set
    return sc_set, unit_set, past_free_set

def _get_last_movement_phase_snapshot(game: Any) -> Tuple[str, Dict[str, Any], Dict[str, List[str]]] | None:
    """
    从 game 的 phase_history 里取“最近一次 Movement phase”的 (phase_name, state_dict, orders_by_power)
    - 兼容不同 dipcc binding：尽量不假设结构，取不到就返回 None
    """
    hist = None
    if hasattr(game, "get_phase_history"):
        try:
            hist = game.get_phase_history()
        except Exception:
            hist = None
    if hist is None and hasattr(game, "phase_history"):
        try:
            hist = getattr(game, "phase_history")
        except Exception:
            hist = None
    if not hist:
        return None

    # 从后往前找最近一次 *M
    for rec in reversed(list(hist)):
        # rec 可能是 dict / obj
        if not isinstance(rec, dict) and hasattr(rec, "to_dict"):
            try:
                rec = rec.to_dict()
            except Exception:
                continue
        if not isinstance(rec, dict):
            continue

        phase_name = str(rec.get("name") or rec.get("phase") or rec.get("phase_name") or "").upper()
        if not phase_name.endswith("M"):
            continue

        st = rec.get("state") or rec.get("game_state") or rec.get("st") or {}
        if not isinstance(st, dict) and hasattr(st, "to_dict"):
            try:
                st = st.to_dict()
            except Exception:
                st = {}
        if not isinstance(st, dict):
            st = {}

        orders = rec.get("orders") or rec.get("orders_by_power") or rec.get("orders_dict") or {}
        # orders 兼容：list[dict] -> dict
        if isinstance(orders, list):
            tmp: Dict[str, List[str]] = {}
            for it in orders:
                if isinstance(it, dict):
                    for k, v in it.items():
                        tmp[str(k)] = list(v) if isinstance(v, (list, tuple)) else [str(v)]
            orders = tmp
        if not isinstance(orders, dict):
            orders = {}

        # 规范 orders: power -> list[str]
        orders_by_power: Dict[str, List[str]] = {}
        for k, v in orders.items():
            if v is None:
                orders_by_power[str(k)] = []
            elif isinstance(v, (list, tuple)):
                orders_by_power[str(k)] = [str(x) for x in v]
            else:
                orders_by_power[str(k)] = [str(v)]

        return phase_name, st, orders_by_power

    return None


def check_c1_intra_turn_consistency(
    game: Any,
    my_power: str,
    action: Any,  # tuple/list[str] or str
) -> Tuple[bool, str]:
    """
    一致性1：当前回合内部一致性（Intra-turn Consistency）
    Conflict if: Helped ∩ Attacked ≠ ∅

    Helped（盟友）：我 Support/Convoy 的其他国家（看 rhs 被帮助单位所在格的占位者）
    Attacked（敌人）：我攻击的国家，或我支援/护送盟友攻击的国家（用各国 Territory 匹配目标地点）
    """

    # --- 取 state（只取一次）---
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

    # --- 预构建：loc -> power（当前占位 unit 的国家）---
    occ = _build_loc2power(game)



    def _dest_base(order_str: str) -> str | None:
        """解析 ' - DEST' 的目标地点，返回 base province（去掉 /SC 等）。"""
        toks = order_str.strip().split()
        if "-" in toks:
            i = toks.index("-")
            if i + 1 < len(toks):
                return str(_norm_loc(toks[i + 1]) or "").split("/")[0] or None
        return None

    

    orders = list(action) if isinstance(action, (list, tuple)) else [str(action)]

    helped: set[str] = set()
    attacked: set[str] = set()

    for od in orders:
        od = str(od)

        # 1) Support/Convoy：解析 rhs
        if " S " in od or " C " in od:
            rhs = od.split(" S ", 1)[1].strip() if " S " in od else od.split(" C ", 1)[1].strip()
            rtoks = rhs.split()

            # Helped：被帮助单位所在格的占位者（其他国家）
            sup_loc = _norm_loc(rtoks[1]) if len(rtoks) >= 2 else None

            # 1) 先拿“被支援单位”的国家（注意：这里要允许拿到 my_power，本函数里不再过滤 self）
            sup_p = None
            if sup_loc:
                sup_p = occ.get(sup_loc)
                if not sup_p and "/" not in sup_loc:
                    cand = [v for k, v in occ.items() if k.startswith(sup_loc + "/")]
                    if cand and all(x == cand[0] for x in cand):
                        sup_p = cand[0]

            # Helped：只记录“我支援了别国”
            if sup_p and sup_p != my_power:
                helped.add(sup_p)

            # 2) 如果 rhs 是 move（带 '-'），再判断“敌人是谁”
            rhs_dest = _dest_base(rhs)
            dest_owner = _owner_by_territory(rhs_dest, st=st, occ=occ, my_power=my_power)



            # ✅ 只有当 dest_owner 存在且 dest_owner != 被支援方国家，才算“攻击 dest_owner”
            #    - dest_owner 为空：空地/不确定 -> 没有敌人
            #    - dest_owner == sup_p：盟友打自己地盘/空地 -> 没有敌人
            if dest_owner and sup_p and dest_owner != sup_p:
                attacked.add(dest_owner)

            

        # 2) 普通 move：Attacked = 目标地点所属国家（Territory 匹配）
        else:
            dest = _dest_base(od)
            dest_owner = _owner_by_territory(dest, st=st, occ=occ, my_power=my_power)
            if dest_owner:
                attacked.add(dest_owner)


    overlap = helped & attacked
    if overlap:
        return False, f"C1_INTRA_TURN: overlap={sorted(overlap)} helped={sorted(helped)} attacked={sorted(attacked)}"
    return True, ""


def check_c2_inter_turn_consistency(
    game: Any,
    my_power: str,
    action: Any,
) -> Tuple[bool, str]:
    """
    一致性2：跨回合战略一致性（Inter-turn Strategic Consistency）

    Last_helped_me（上轮帮助我的国家）：
      - 上一轮(最近一次春/秋M) 其他国家 Support/Convoy 我方 unit（看 rhs 被帮助单位所在格的占位者 == my_power）

    Last_attacked_me（上轮攻击我的国家）：
      - 上一轮(最近一次春/秋M) 其他国家存在 Move 或 SupportMove，其目标地点落在我方 Territory（Territory=SC+unit+past_free）
      - 默认不把 Convoy 计为“攻击我”（你确认要不要算）

    约束：
      - 对 Last_helped_me：本轮不攻击（不进入 Attacked 集合）
      - 对 Last_attacked_me：本轮不支援（不进入 Helped 集合）
    """
    # 只在 Movement phase 生效（默认）
    try:
        cur_phase = str(game.get_current_phase()).upper()
    except Exception:
        cur_phase = ""
    if not cur_phase.endswith("M"):
        return True, ""

    snap = _get_last_movement_phase_snapshot(game)
    if not snap:
        return True, ""  # 没历史就不做过滤
    prev_phase, st_prev, prev_orders = snap

    # ====== 上一轮：构造 occ_prev（loc->power）+ 我方 Territory(prev) ======
    units_prev = st_prev.get("units", {}) or {}
    occ_prev: Dict[str, str] = {}
    if isinstance(units_prev, dict):
        for pwr, ulist in units_prev.items():
            for u in (ulist or []):
                parts = str(u).strip().lstrip("*").split()
                if len(parts) >= 2:
                    loc = _norm_loc(parts[1])
                    if loc:
                        occ_prev[loc] = str(pwr)

    sc_set, unit_set, past_free_set = get_territory_parts(st_prev, my_power)
    my_terr_prev = sc_set | unit_set | past_free_set  # base province

    def _owner_by_occupy_prev(loc: str | None) -> str | None:
        nl = _norm_loc(loc)
        if not nl:
            return None
        p = occ_prev.get(nl)
        if not p and "/" not in nl:
            cand = [v for k, v in occ_prev.items() if k.startswith(nl + "/")]
            if cand and all(x == cand[0] for x in cand):
                p = cand[0]
        return p

    def _dest_base(order_str: str) -> str | None:
        toks = order_str.strip().split()
        if "-" in toks:
            i = toks.index("-")
            if i + 1 < len(toks):
                return str(_norm_loc(toks[i + 1]) or "").split("/")[0] or None
        return None

    # ====== 计算 Last_helped_me / Last_attacked_me ======
    last_helped_me: set[str] = set()
    last_attacked_me: set[str] = set()

    for pwr, olist in (prev_orders or {}).items():
        pwr = str(pwr)
        if pwr == my_power:
            continue
        for od in (olist or []):
            od = str(od)

            # (A) 统计谁帮过我：Support/Convoy 我方 unit
            if " S " in od or " C " in od:
                rhs = od.split(" S ", 1)[1].strip() if " S " in od else od.split(" C ", 1)[1].strip()
                rtoks = rhs.split()
                sup_loc = _norm_loc(rtoks[1]) if len(rtoks) >= 2 else None
                sup_owner = _owner_by_occupy_prev(sup_loc)
                if sup_owner == my_power:
                    last_helped_me.add(pwr)

            # (B) 统计谁攻击过我：Move 或 SupportMove 目标落入我方 Territory(prev)
            if " S " in od:
                rhs = od.split(" S ", 1)[1].strip()
                rtoks = rhs.split()
                sup_loc = _norm_loc(rtoks[1]) if len(rtoks) >= 2 else None
                sup_owner = _owner_by_occupy_prev(sup_loc)  # ✅ 被支援的单位属于谁

                # ✅ 只有“支援他国单位”打进我方 Territory 才算攻击
                if "-" in rhs:
                    dest = _dest_base(rhs)
                    if dest and dest in my_terr_prev and sup_owner != my_power:
                        last_attacked_me.add(pwr)
            else:
                # 普通 move（含 '-'；这里 convoy 也会落到这条，符合你“也算敌对”的口径）
                dest = _dest_base(od)
                if dest and dest in my_terr_prev:
                    last_attacked_me.add(pwr)


    if not last_helped_me and not last_attacked_me:
        return True, ""

    # ====== 当前候选 action：计算 Helped_now / Attacked_now（都按 Territory 匹配） ======
    try:
        st_cur = game.get_state()
    except Exception:
        st_cur = {}
    if not isinstance(st_cur, dict) and hasattr(st_cur, "to_dict"):
        try:
            st_cur = st_cur.to_dict()
        except Exception:
            st_cur = {}
    if not isinstance(st_cur, dict):
        st_cur = {}

    occ_cur = _build_loc2power(game)

    # terr_by_pwr_cur: Dict[str, set[str]] = {}
    # for pwr in POWERS:
    #     sc, uu, pf = get_territory_parts(st_cur, pwr)
    #     terr_by_pwr_cur[pwr] = sc | uu | pf

    # def _owner_by_occupy_cur(loc: str | None) -> str | None:
    #     nl = _norm_loc(loc)
    #     if not nl:
    #         return None
    #     p = occ_cur.get(nl)
    #     if not p and "/" not in nl:
    #         cand = [v for k, v in occ_cur.items() if k.startswith(nl + "/")]
    #         if cand and all(x == cand[0] for x in cand):
    #             p = cand[0]
    #     if not p or p == my_power:
    #         return None
    #     return p

    # def _owner_by_territory_cur(prov_base: str | None) -> str | None:
    #     if not prov_base:
    #         return None
    #     owners = [pwr for pwr in POWERS if prov_base in terr_by_pwr_cur.get(pwr, set())]
    #     if len(owners) != 1:
    #         return None
    #     return None if owners[0] == my_power else owners[0]

    orders_now = list(action) if isinstance(action, (list, tuple)) else [str(action)]
    helped_now: set[str] = set()
    attacked_now: set[str] = set()

    for od in orders_now:
        od = str(od)

        if " S " in od or " C " in od:
            rhs = od.split(" S ", 1)[1].strip() if " S " in od else od.split(" C ", 1)[1].strip()
            rtoks = rhs.split()

            # Helped_now：被帮助单位占位者
            sup_loc = _norm_loc(rtoks[1]) if len(rtoks) >= 2 else None
            # 先拿被支援单位国家（允许是 my_power）
            sup_p = None
            if sup_loc:
                sup_p = occ_cur.get(sup_loc)
                if not sup_p and "/" not in sup_loc:
                    cand = [v for k, v in occ_cur.items() if k.startswith(sup_loc + "/")]
                    if cand and all(x == cand[0] for x in cand):
                        sup_p = cand[0]

            if sup_p and sup_p != my_power:
                helped_now.add(sup_p)

            rhs_dest = _dest_base(rhs)
            dest_owner = _owner_by_territory(rhs_dest, st=st_cur, occ=occ_cur, my_power=my_power)

            if dest_owner and sup_p and dest_owner != sup_p:
                attacked_now.add(dest_owner)

        else:
            dest = _dest_base(od)
            dest_owner = _owner_by_territory(dest, st=st_cur, occ=occ_cur, my_power=my_power)
            if dest_owner:
                attacked_now.add(dest_owner)


    bad_attack_helper = attacked_now & last_helped_me
    bad_help_attacker = helped_now & last_attacked_me

    if bad_attack_helper or bad_help_attacker:
        parts = []
        if bad_attack_helper:
            parts.append(
                f"attacked_last_helper={sorted(bad_attack_helper)} "
                f"last_helped_me={sorted(last_helped_me)} attacked_now={sorted(attacked_now)} prev={prev_phase}"
            )
        if bad_help_attacker:
            parts.append(
                f"helped_last_attacker={sorted(bad_help_attacker)} "
                f"last_attacked_me={sorted(last_attacked_me)} helped_now={sorted(helped_now)} prev={prev_phase}"
            )
        return False, "C2_INTER_TURN: " + " | ".join(parts)

    return True, ""


def check_c3_destination_conflict(
    game: Any,
    my_power: str,
    action: Any,
) -> Tuple[bool, str]:
    """
    一致性3：目标地点冲突（Destination Conflict）

    规则：
    1) 我支援/护送“盟友”去的目的地 ∩ 我自己单位去的目的地 = ∅
       - 这里“盟友”指被 support/convoy 的单位不属于我方（用占位 occ 来判断）
    2) 对同一目标地点，最多只有一个我的单位 Move(含 VIA) 过去；否则应该用 Support
       - 这里“Convoy过去”理解为：被 convoy 的陆军仍然是 Move 指令（A X - Y VIA），所以统计在 Move 里即可
    """
    # ✅ 只在 Movement phase 做检查（避免 retreat/build 的语法干扰）
    try:
        cur_phase = str(game.get_current_phase()).upper()
    except Exception:
        cur_phase = ""
    if not cur_phase.endswith("M"):
        return True, ""

    orders = list(action) if isinstance(action, (list, tuple)) else [str(action)]

    # 当前占位：loc(with coast) -> power
    occ = _build_loc2power(game)

    # 小工具：用 occ 查某个 loc 当前是谁的单位（支持 STP vs STP/SC 兜底）
    def _owner_by_occupy(loc: str | None) -> str | None:
        nl = _norm_loc(loc)
        if not nl:
            return None
        p = occ.get(nl)
        if not p and "/" not in nl:
            cand = [v for k, v in occ.items() if k.startswith(nl + "/")]
            if cand and all(x == cand[0] for x in cand):
                p = cand[0]
        return p

    # 记录：我“帮盟友去”的目的地、我“自己去”的目的地（都用 base province）
    helped_ally_dest2orders: Dict[str, List[str]] = {}
    my_dest2orders: Dict[str, List[str]] = {}

    for od in orders:
        s = str(od).strip()
        if not s:
            continue

        # ---------- (A) Support / Convoy：解析 rhs ----------
        if " S " in s or " C " in s:
            rhs = s.split(" S ", 1)[1].strip() if " S " in s else s.split(" C ", 1)[1].strip()
            rtoks = rhs.split()

            # rhs 的被帮助单位所在格（rtoks[1]）
            sup_loc = _norm_loc(rtoks[1]) if len(rtoks) >= 2 else None
            sup_owner = _owner_by_occupy(sup_loc)

            # 只关心 support/convoy 的 rhs 是一个 move（带 '-'）的情况
            toks = rhs.split()
            if "-" in toks:
                i = toks.index("-")
                if i + 1 < len(toks):
                    dest = str(_norm_loc(toks[i + 1]) or "").split("/")[0]

                    # ✅ 只把“帮盟友”的目的地纳入冲突检测（帮自己不算）
                    if dest and sup_owner and sup_owner != my_power:
                        helped_ally_dest2orders.setdefault(dest, []).append(s)

        # ---------- (B) 我自己的 Move(含 VIA)：解析目的地 ----------
        else:
            toks = s.split()
            if "-" in toks:
                i = toks.index("-")
                if i + 1 < len(toks):
                    dest = str(_norm_loc(toks[i + 1]) or "").split("/")[0]
                    if dest:
                        my_dest2orders.setdefault(dest, []).append(s)

    # 规则1：帮盟友去的地，不能和我自己去的地重合
    overlap = set(helped_ally_dest2orders.keys()) & set(my_dest2orders.keys())

    # 规则2：同一目的地，最多一个我方 Move
    multi_move = {d: ods for d, ods in my_dest2orders.items() if len(ods) > 1}

    if overlap or multi_move:
        parts: List[str] = []

        if overlap:
            # 只列出冲突目的地，附上相关 orders（便于你 debug）
            detail = []
            for d in sorted(overlap):
                detail.append(
                    f"{d}: help={helped_ally_dest2orders.get(d, [])} | move={my_dest2orders.get(d, [])}"
                )
            parts.append("help∩move=" + "; ".join(detail))

        if multi_move:
            detail = []
            for d in sorted(multi_move.keys()):
                detail.append(f"{d}: moves={multi_move[d]}")
            parts.append("multi_move=" + "; ".join(detail))

        return False, "C3_DEST_CONFLICT: " + " | ".join(parts)

    return True, ""


def check_c4_self_defense_consistency(
    game: Any,
    my_power: str,
    action: Any,
) -> Tuple[bool, str]:
    """
    一致性4：自我防御一致性（含你新增的放行条件）
    规则：
      - 默认：Support/Convoy 别国单位去我方 Territory -> 过滤
      - 例外（仅 nonSC）：若目的地没有我方unit，或该unit本回合会从该地移走 -> 放行
      - SC 仍严格保护（即使本回合要移走也不过）
    """
    try:
        cur_phase = str(game.get_current_phase()).upper()
    except Exception:
        cur_phase = ""
    if not cur_phase.endswith("M"):
        return True, ""

    # 当前 state + 我方 territory
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

    sc_set, unit_set, past_free_set = get_territory_parts(st, my_power)
    my_terr = sc_set | unit_set | past_free_set  # base province

    # 当前占位（判断被支援单位属于谁）
    occ = _build_loc2power(game)

    def _owner_by_occupy(loc: str | None) -> str | None:
        nl = _norm_loc(loc)
        if not nl:
            return None
        p = occ.get(nl)
        if not p and "/" not in nl:
            cand = [v for k, v in occ.items() if k.startswith(nl + "/")]
            if cand and all(x == cand[0] for x in cand):
                p = cand[0]
        return p

    # 统计“我方本回合将从哪些base省份移走”（仅普通 move，不含 S/C）
    orders = list(action) if isinstance(action, (list, tuple)) else [str(action)]
    my_move_from: set[str] = set()
    for od in orders:
        s = str(od).strip()
        if not s or " S " in s or " C " in s:
            continue
        toks = s.split()
        if "-" in toks and len(toks) >= 2:
            src_base = str(_norm_loc(toks[1]) or "").split("/")[0]
            if src_base:
                my_move_from.add(src_base)

    bad: List[str] = []

    for od in orders:
        s = str(od).strip()
        if not s:
            continue
        if " S " not in s and " C " not in s:
            continue

        rhs = s.split(" S ", 1)[1].strip() if " S " in s else s.split(" C ", 1)[1].strip()
        rtoks = rhs.split()

        # 被帮助单位是谁（只看“帮别国”）
        sup_loc = _norm_loc(rtoks[1]) if len(rtoks) >= 2 else None
        sup_owner = _owner_by_occupy(sup_loc)
        if not sup_owner or sup_owner == my_power:
            continue

        # rhs 不是 move 就跳过
        toks = rhs.split()
        if "-" not in toks:
            continue
        i = toks.index("-")
        if i + 1 >= len(toks):
            continue

        dest_base = str(_norm_loc(toks[i + 1]) or "").split("/")[0]
        if not dest_base or dest_base not in my_terr:
            continue

        # SC：严格拦截
        if dest_base in sc_set:
            bad.append(f"dest={dest_base}(SC) rhs='{rhs}' order='{s}'")
            continue

        # nonSC：你新增的放行条件
        # 1) 该地当前无我方unit；或 2) 我方unit本回合会从该地移走
        if (dest_base not in unit_set) or (dest_base in my_move_from):
            continue

        # nonSC 且我方unit留守 -> 拦截
        bad.append(f"dest={dest_base}(nonSC_hold) rhs='{rhs}' order='{s}'")

    if bad:
        return False, "C4_SELF_DEFENSE: " + " | ".join(bad)
    return True, ""



def filter_action_set_by_consistency(
    game: Any,
    my_power: str,
    items: List[Tuple[Any, float]],
) -> Tuple[List[Tuple[Any, float]], List[Tuple[Any, float, str]]]:
    """
    统一过滤器：按 4 个一致性依次过滤 action set
    - 本次只启用一致性1，其余一致性函数先占位不做
    - dropped 的 reason 必须带一致性类型，便于 LOG 标注
    """
    kept: List[Tuple[Any, float]] = []
    dropped: List[Tuple[Any, float, str]] = []

    for action, p in items:
        ok, reason = check_c1_intra_turn_consistency(game, my_power, action)
        if not ok:
            dropped.append((action, p, reason))
            continue

        ok2, reason2 = check_c2_inter_turn_consistency(game, my_power, action)
        if not ok2:
            dropped.append((action, p, reason2))
            continue


        ok3, r3 = check_c3_destination_conflict(game, my_power, action)
        if not ok3:
            dropped.append((action, p, r3))
            continue

        ok4, r4 = check_c4_self_defense_consistency(game, my_power, action)
        if not ok4:
            dropped.append((action, p, r4))
            continue


        kept.append((action, p))

    return kept, dropped




def main():
    """
    用 consistent_agent 跑一个 dipcc game：
    - 每个 phase：对指定 power（例如 AUSTRIA）调用 choose_orders 拿到 top-k items
    - 把 items（p, action）写到 log，便于你核对候选动作列表是否正确
    - 其他国家用 blueprint 的 top1（或随机）补齐 orders，保证 game.process() 能跑
    输出：
      - 生成一个 .log 文件，包含每步 phase、source、topk actions
    """
    import argparse
    import os
    from datetime import datetime
    from fairdiplomacy import pydipcc

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="conf/common/agents/consistent_agent.prototxt")
    parser.add_argument("--project_root", type=str, default="/workspace/Diplomacy/diplomacy_cicero")
    parser.add_argument("--power", type=str, default="AUSTRIA", choices=POWERS)
    parser.add_argument("--seed", type=int, default=0)

    # ✅ 修改：默认走 bqre_topK，并把 choices 加上 bqre_topK
    parser.add_argument("--source", type=str, default="bqre_topK",
                        choices=["bqre_topK", "search_br", "bp"])
    parser.add_argument("--mode", type=str, default="top1", choices=["top1", "sample"])
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max_phases", type=int, default=60)

    parser.add_argument("--log_dir", type=str, default="logs_consistent")
    parser.add_argument("--log", type=str, default=None)

    args = parser.parse_args()
    random.seed(args.seed)
    # ====== 全局可复现：尽量把所有随机源都固定住 ======
    import os
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    try:
        import numpy as np
        np.random.seed(args.seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # 尽量 deterministic（可能牺牲一点速度）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


    # 1) 进入项目根目录（保证相对路径的模型/配置能找到）
    if args.project_root and os.path.exists(args.project_root):
        os.chdir(args.project_root)

    # 2) log 路径
    ts = datetime.now().strftime("%y%m%d%H%M%S")
    log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(os.getcwd(), args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = args.log if args.log else os.path.join(log_dir, f"consistent_{ts}.log")

    # 3) 加载 agent + 初始化 game/state
    agent = load_cicero(args.cfg, skip_cache=False)
    game = pydipcc.Game()
    states = {p: agent.initialize_state(p) for p in POWERS}

    def _is_done(g: "pydipcc.Game") -> bool:
        # 尽量兼容不同 dipcc binding
        for attr in ("is_game_done", "is_game_over", "game_over"):
            if hasattr(g, attr):
                try:
                    v = getattr(g, attr)
                    return bool(v() if callable(v) else v)
                except Exception:
                    pass
        ph = str(g.get_current_phase()).upper()
        return ("COMPLETED" in ph) or (ph in {"DONE", "END"})

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== CONSISTENT_AGENT RUN START ===\n")
        f.write(f"cwd={os.getcwd()}\n")
        f.write(f"cfg={args.cfg}\n")
        f.write(f"power={args.power}, seed={args.seed}, source={args.source}, mode={args.mode}, topk={args.topk}\n\n")
        f.flush()

        step = 0
        while step < args.max_phases and not _is_done(game):
            phase = game.get_current_phase()

             # ✅ 新增：每回合开始抓一次 state（用于 log: units / SC / territory）
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
            # 【七个国家都是bqre_topK】
            # --- 给所有国家补齐 orders，保证能 process ---
            set_orders: Dict[str, List[str]] = {p: [] for p in POWERS}

            # 先让所有国家“在同一个 state 下”各自选单：只存，不写入 game（防止信息泄露）
            all_infos: Dict[str, Tuple[List[Tuple[Any, float]], str, List[Tuple[Any, float, str]]]] = {}
            tmp_orders: Dict[str, List[str]] = {}

            for pwr in POWERS:
                orders, items, used_source, dropped = agent.choose_orders(
                    game=game,
                    power=pwr,
                    agent_state=states[pwr],
                    source=args.source,
                    top_k=args.topk,
                    mode=args.mode,
                )
                tmp_orders[pwr] = orders
                all_infos[pwr] = (items, used_source, dropped)

            f.write("\n" + "=" * 90 + "\n")
            f.write(f"[STEP {step:04d}] phase={phase}\n")

            # ✅ 每回合开始时打印所有玩家 units / SC / nonSC territory（你原来的逻辑保留）
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

            # ✅ 打印 7 国各自的 FILTERED OUT（以及可选的 topk 列表）
            for pwr in POWERS:
                items, used_source, dropped = all_infos[pwr]
                f.write(f"[AGENT] power={pwr} used_source={used_source} topk={len(items)}\n")

                f.write(f"[FILTERED OUT] power={pwr} n={len(dropped)}\n")
                for j, (a, pp, rsn) in enumerate(dropped):
                    if isinstance(a, (list, tuple)):
                        act_str = "[" + ", ".join(map(str, a)) + "]"
                    else:
                        act_str = str(a)
                    f.write(f"  -{j:02d}  p={float(pp):.8f}  reason={rsn}  action={act_str}\n")

                # 如果你也想每个国家都打印 kept 的 topk（会很长），取消注释：
                # for i, (a, p) in enumerate(items):
                #     act_str = "[" + ", ".join(map(str, a)) + "]" if isinstance(a, (list, tuple)) else str(a)
                #     f.write(f"  #{i:02d}  p={float(p):.8f}  action={act_str}\n")

            # ✅ 最后一次性写入 orders（避免后选国家“看见”先选国家 orders）
            for pwr in POWERS:
                set_orders[pwr] = tmp_orders.get(pwr, [])
                game.set_orders(pwr, set_orders[pwr])

            f.write("[ORDERS SET]\n")
            for pwr in POWERS:
                f.write(f"  {pwr}: {set_orders[pwr]}\n")

            # # 【其他国家用BP】
            # # --- 给所有国家补齐 orders，保证能 process --- 
            # set_orders: Dict[str, List[str]] = {p: [] for p in POWERS}

            # # 我方：拿 top-k items 并写 log（默认 bqre_topK）
            # my_orders, my_items, used_source, my_dropped = agent.choose_orders(
            #     game=game,
            #     power=args.power,
            #     agent_state=states[args.power],
            #     source=args.source,
            #     top_k=args.topk,
            #     mode=args.mode,
            # )
            # set_orders[args.power] = my_orders
            # game.set_orders(args.power, my_orders)

            # f.write("\n" + "=" * 90 + "\n")
            # f.write(f"[STEP {step:04d}] phase={phase}\n")

            # # ✅ 新增：每回合开始时打印所有玩家 units / SC / nonSC territory
            # units = st.get("units", {}) or {}
            # centers = st.get("centers", {}) or {}
            # influence = st.get("influence", None)
            # terr_src = "influence" if isinstance(influence, dict) else "fallback"

            # f.write(f"[STATE BEFORE] terr_src={terr_src}\n")
            # for pwr in POWERS:
            #     ulist = list((units.get(pwr) or []))
            #     sclist_raw = list((centers.get(pwr) or []))

            #     sc_set, unit_set, past_free_set = get_territory_parts(st, pwr)
            #     terr_set = sc_set | unit_set | past_free_set  # 这就是你定义的 Territory
            #     sc_list = sorted(sc_set)
            #     non_sc = sorted(terr_set - sc_set)


            #     f.write(
            #         f"  {pwr}: "
            #         f"units({len(ulist)})={ulist} | "
            #         f"SC({len(sc_list)})={sc_list} | "
            #         f"nonSC({len(non_sc)})={non_sc}\n"
            #     )

            # f.write(f"[ME] power={args.power} used_source={used_source} topk={len(my_items)}\n")

            # # ---- #
            # # ✅ 过滤信息输出（后续你可以整段注释掉）
            # f.write(f"[FILTERED OUT] n={len(my_dropped)}\n")
            # for j, (a, pp, rsn) in enumerate(my_dropped):
            #     if isinstance(a, (list, tuple)):
            #         act_str = "[" + ", ".join(map(str, a)) + "]"
            #     else:
            #         act_str = str(a)
            #     f.write(f"  -{j:02d}  p={float(pp):.8f}  reason={rsn}  action={act_str}\n")
            # # ---- #

            # for i, (a, p) in enumerate(my_items):
            #     # a 通常是 tuple/list[str]（一组 orders）
            #     if isinstance(a, (list, tuple)):
            #         act_str = "[" + ", ".join(map(str, a)) + "]"
            #     else:
            #         act_str = str(a)
            #     f.write(f"  #{i:02d}  p={float(p):.8f}  action={act_str}\n")

            # # 其他国家：用 bp 的 top1（极简兜底）
            # for pwr in POWERS:
            #     if pwr == args.power:
            #         continue

            #     bp_pol = agent.get_plausible_orders_policy(game=game, agent_power=pwr, agent_state=states[pwr])
            #     dist = bp_pol.get(pwr, {}) or {}
            #     if dist:
            #         items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
            #         action = items[0][0]
            #         orders = list(action) if isinstance(action, (list, tuple)) else [action]
            #     else:
            #         orders = []

            #     set_orders[pwr] = orders
            #     game.set_orders(pwr, orders)

            # f.write("[ORDERS SET]\n")
            # for pwr in POWERS:
            #     f.write(f"  {pwr}: {set_orders[pwr]}\n")


            #----#

            # 推进一回合
            try:
                game.process()
            except Exception as e:
                f.write(f"[ERROR] game.process() failed @phase={phase}: {repr(e)}\n")
                break

            f.flush()
            step += 1

        f.write("\n=== RUN END ===\n")
        f.write(f"final_phase={game.get_current_phase()}\n")
        f.flush()

    print(f"[OK] log saved to: {log_path}")


if __name__ == "__main__":
    main()
