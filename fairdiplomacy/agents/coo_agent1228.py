# coding=utf-8
from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Tuple
from datetime import datetime

import heyhi
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent
from fairdiplomacy.agents.coo_coalition import CoalitionSolver, CoalitionSolverConfig
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
phase_step = 0
movement_step = 0

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



# ---------------------------
# Main loop: run-to-end + log.txt (UPDATED)
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="conf/common/agents/coo_agent.prototxt")
    parser.add_argument("--project_root", type=str, default="/workspace/Diplomacy/diplomacy_cicero")
    parser.add_argument("--power", type=str, default="FRANCE", choices=POWERS)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", type=str, default="search_br", choices=["search_br", "bp"])
    parser.add_argument("--mode", type=str, default="sample", choices=["sample", "top1"])
    parser.add_argument("--topk", type=int, default=30)

    parser.add_argument("--others", type=str, default="base", choices=["base", "hold"])

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

    self_agent = load_cicero(args.cfg, skip_cache=False)

    game = pydipcc.Game()
    self_state = self_agent.initialize_state(args.power)

    prev_sc_owner = None  # for SC diff
    step = 0

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== RUN START ===\n")
        f.write(f"cwd={os.getcwd()}\n")
        f.write(f"cfg={args.cfg}\n")
        f.write(
            f"power={args.power}, seed={args.seed}, source={args.source}, mode={args.mode}, "
            f"topk={args.topk}, others={args.others}, max_phases={args.max_phases}\n\n"
        )
        f.flush()
        coalition_solver = CoalitionSolver(CoalitionSolverConfig(top_k=4, min_coalition_size=1, max_coalition_size=2))
        last_set_orders = {}  # optional: 给 H_t 占位用
        movement_step = 0
        phase_step = 0

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
            


            # --- policy ---
            bp_policy: Dict[str, Dict[Any, float]] = self_agent.get_plausible_orders_policy(
                game=game,
                agent_power=args.power,
                agent_state=self_state,
            )

            H_t = {"last_orders": last_set_orders}  # 占位 History
            g = coalition_solver.solve_coalition(
                game=game,
                me=args.power,
                pi_me=args.source,     # 占位
                pi_bp=bp_policy,       # 先给进去，后面 rollout 会用
                H_t=H_t,
                step=movement_step,
            )

            # 写入同一个 log 文件（候选盟友来自 E）
            cands = [j for (_, j) in g.E]
            cand_str = ", ".join([f"{j}(EU={g.w[(args.power, j)]:.2f})" for j in cands])
            f.write(
                f"[ALLIANCE] phase_step={phase_step} movement_step={movement_step} phase={g.meta['phase']} "
                f"candidates=[{cand_str}] chosen={g.chosen_allies}\n"
            )


            used_source = "bp"
            my_dist = bp_policy.get(args.power, {}) or {}

            # search_br only for movement
            if args.debug:
                print(f"\n[DBG] phase={phase} label={meta.get('label')} kind={meta.get('kind')} is_movement={is_movement} args.source={args.source}")
                print(f"[DBG] will_try_search_br? {args.source=='search_br' and is_movement}")
            if args.pause:
                input("[DBG] enter to continue ... ")

            if args.source == "search_br" and is_movement:
                try:
                    if args.debug:
                        print("[DBG] running run_best_response_against_correlated_bilateral_search(...)")
                    search_res = self_agent.run_best_response_against_correlated_bilateral_search(
                        game=game,
                        agent_power=args.power,
                        bp_policy=bp_policy,
                        agent_state=self_state,
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
            elif args.source == "search_br" and (not is_movement):
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
            my_orders = _pick_orders_from_dist(my_dist, top_k=args.topk, mode=args.mode)

            # --- set orders for all powers ---
            set_orders: Dict[str, List[str]] = {p: [] for p in POWERS}

            set_orders[args.power] = my_orders
            game.set_orders(args.power, my_orders)

            for p in POWERS:
                if p == args.power:
                    continue
                if args.others == "hold":
                    o = []  # NOTE: if this ever causes process() errors, we can generate explicit HOLD orders.
                else:
                    o = _pick_orders_from_dist(bp_policy.get(p, {}) or {}, top_k=args.topk, mode="top1")
                set_orders[p] = o
                game.set_orders(p, o)

            # --- log BEFORE process ---
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"[PHASE {step:04d}] phase={phase} | {meta.get('label')}\n")
            f.write(f"[PHASE META] kind={meta.get('kind')} season={meta.get('season')} year={meta.get('year')}\n")

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

            f.write(f"[MY ACTION DIST] used_source={used_source}, topk={len(my_items)}\n")
            f.write(_format_action_dist(my_items) + "\n")

            f.write("[ORDERS SET]\n")
            for p in POWERS:
                f.write(f"  {p}: {set_orders[p]}\n")

            # --- advance ---
            try:
                game.process()
            except Exception as e:
                f.write(f"[ERROR] game.process() failed @phase={phase} ({meta.get('label')}): {repr(e)}\n")
                break
            if is_movement:
                movement_step += 1
            phase_step += 1

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

        f.write(f"\nfinal_phase={game.get_current_phase()}\n")
        f.write("=== RUN END ===\n")
        f.flush()

if __name__ == "__main__":
    main()
