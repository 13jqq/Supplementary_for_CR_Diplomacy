# coding=utf-8
from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import heyhi
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent
from fairdiplomacy.utils.sampling import sample_p_dict

POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

Action = Any
CandidateItems = List[Tuple[Action, float]]


class CR2Agent(BQRE1PAgent):
    """
    CR^2 decision agent.

    The agent first obtains a candidate action distribution from an underlying
    policy or search module. It then evaluates each candidate along structural
    dimensions and converts the resulting augmented utility into the final
    sampling distribution.

    Public interface:
      - get_orders(...): returns the final order list.
      - get_orders_info(...): returns the final order list and candidate-level
        diagnostic information for logging.
    """

    CHECK_ORDER = ("C1", "C2", "C3", "C4")

    def get_orders_info(
        self,
        game: Any,
        power: str,
        state: Any,
        *,
        source: str = "bqre_topK",
        top_k: int = 30,
        mode: str = "bqre",
    ) -> Dict[str, Any]:
        """
        Build a CR^2 action distribution and sample the final action.

        Returns:
        {
            "orders": List[str],
            "items": List[(action, prob)],
            "raw_items": List[(action, prob)],
            "used_source": str,
            "dropped": List[(action, prob, reason)],
            "c3a_logs": List[Dict[str, Any]],
            "structural_metrics": List[Dict[str, Any]],
        }

        The field "dropped" is kept for compatibility with existing runners.
        CR^2 uses structural regularization rather than hard candidate removal,
        so it is normally empty in this version.
        """
        bp_policy: Dict[str, Dict[Any, float]] = self.get_plausible_orders_policy(
            game=game,
            agent_power=power,
            agent_state=state,
        )
        dist: Dict[Any, float] = bp_policy.get(power, {}) or {}
        used_source = "bp"

        if source == "bqre_topK":
            res = self.run_search(
                game=game,
                bp_policy=bp_policy,
                agent_power=power,
                agent_state=state,
            )
            dist = res.get_agent_policy().get(power, {}) or {}
            used_source = "bqre_topK"
        elif source == "search_br":
            search_res = self.run_best_response_against_correlated_bilateral_search(
                game=game,
                agent_power=power,
                bp_policy=bp_policy,
                agent_state=state,
            )
            agent_pols = search_res.get_agent_policy()
            if agent_pols.get(power):
                dist = agent_pols[power]
                used_source = "search_br"

        if not dist:
            return {
                "orders": [],
                "items": [],
                "raw_items": [],
                "used_source": used_source,
                "dropped": [],
                "c3a_logs": [],
                "structural_metrics": [],
            }

        raw_items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None and top_k > 0:
            raw_items = raw_items[:top_k]

        items, structural_metrics = regularize_action_set_by_structure(
            game=game,
            my_power=power,
            items=raw_items,
        )

        action = self._select_action_from_items(items, mode=mode)
        orders = list(action) if isinstance(action, (list, tuple)) else [action]

        return {
            "orders": orders,
            "items": items,
            "raw_items": raw_items,
            "used_source": used_source,
            "dropped": [],
            "c3a_logs": [],
            "structural_metrics": structural_metrics,
        }

    def get_orders(
        self,
        game: Any,
        power: str,
        state: Any,
        *,
        source: str = "bqre_topK",
        top_k: int = 30,
        mode: str = "bqre",
    ) -> List[str]:
        """Return only the final order list required by the agent interface."""
        info = self.get_orders_info(
            game=game,
            power=power,
            state=state,
            source=source,
            top_k=top_k,
            mode=mode,
        )
        return info["orders"]

    def _select_action_from_items(
        self,
        items: CandidateItems,
        *,
        mode: str = "bqre",
    ) -> Any:
        """Select an action from a regularized candidate distribution."""
        if not items:
            return []

        if mode == "sample":
            dist = _sharpen_action_items(items, beta=2.0)
            if dist is None:
                return random.choice([a for a, _ in items])
            return sample_p_dict(dist)

        if mode == "bqre":
            dist = _renorm_action_items(items)
            if dist is None:
                return random.choice([a for a, _ in items])
            return sample_p_dict(dist)

        return max(items, key=lambda kv: kv[1])[0]

    def audit_final_orders(
        self,
        game: Any,
        power: str,
        orders: Any,
        *,
        check_order: Tuple[str, ...] = CHECK_ORDER,
    ) -> Dict[str, Any]:
        """
        Evaluate the structural status of a finalized action.

        Returns:
        {
            "power": str,
            "ok": bool,
            "violations": List[(tag, reason)],
        }
        """
        action = self._normalize_action(orders)
        checks = {
            "C1": check_c1_intra_turn_consistency,
            "C2": check_c2_inter_turn_consistency,
            "C3": check_c3_destination_conflict,
            "C4": check_c4_self_defense_consistency,
        }

        violations: List[Tuple[str, str]] = []
        for tag in check_order:
            fn = checks.get(tag)
            if fn is None:
                continue
            ok, reason = fn(game, power, action)
            if not ok:
                violations.append((tag, reason))

        return {
            "power": power,
            "ok": len(violations) == 0,
            "violations": violations,
        }

    @staticmethod
    def _normalize_action(orders: Any) -> Tuple[str, ...]:
        """Convert an order representation into a tuple of order strings."""
        if orders is None:
            return tuple()
        if isinstance(orders, tuple):
            return tuple(str(x) for x in orders)
        if isinstance(orders, list):
            return tuple(str(x) for x in orders)
        if isinstance(orders, str):
            return (orders,)
        try:
            return tuple(str(x) for x in orders)
        except Exception:
            return (str(orders),)


class CR2DocusAgent(CR2Agent):
    """
    CR^2 variant using the Diplodocus-style BQRE policy object.

    This class keeps the native policy extraction routine while using the same
    structural regularization and sampling interface as CR2Agent.
    """

    def _get_native_bqre_policy_from_result(self, res: Any, power: str) -> Dict[Any, float]:
        ptype_policies = res.ptype_final_policies if res.use_final_iter else res.ptype_avg_policies
        return ptype_policies[res.agent_type].get(power, {}) or {}

    def _sample_native_bqre_action_with_structure(
        self,
        *,
        res: Any,
        game: Any,
        power: str,
    ) -> Any:
        policy = self._get_native_bqre_policy_from_result(res, power)
        if not policy:
            return []
        raw_items = sorted(policy.items(), key=lambda kv: kv[1], reverse=True)
        items, _ = regularize_action_set_by_structure(game=game, my_power=power, items=raw_items)
        dist = _renorm_action_items(items)
        if dist is None:
            return random.choice([a for a, _ in raw_items])
        return sample_p_dict(dist)


class ConsistentAgent(CR2Agent):
    """Backward-compatible class name for existing configuration files."""


class ConsistentDocusAgent(CR2DocusAgent):
    """Backward-compatible class name for existing configuration files."""


def load_consistent_agent(cfg_path: str, *, skip_cache: bool = False) -> ConsistentAgent:
    """Load a CR^2 agent from a consistent_agent configuration block."""
    full_cfg = heyhi.load_config(cfg_path)
    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "consistent_agent"):
        agent_cfg = full_cfg.agent.consistent_agent
    elif hasattr(full_cfg, "consistent_agent"):
        agent_cfg = full_cfg.consistent_agent
    else:
        raise ValueError(f"Bad config structure in {cfg_path}: cannot find consistent_agent")
    return ConsistentAgent(agent_cfg, skip_base_strategy_model_cache=skip_cache)


def load_consistent_docus_agent(cfg_path: str, *, skip_cache: bool = False) -> ConsistentDocusAgent:
    """Load the Diplodocus-style CR^2 agent from a consistent_agent block."""
    full_cfg = heyhi.load_config(cfg_path)
    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "consistent_agent"):
        agent_cfg = full_cfg.agent.consistent_agent
    elif hasattr(full_cfg, "consistent_agent"):
        agent_cfg = full_cfg.consistent_agent
    else:
        raise ValueError(f"Bad config structure in {cfg_path}: cannot find consistent_agent")
    return ConsistentDocusAgent(agent_cfg, skip_base_strategy_model_cache=skip_cache)


def _renorm_action_items(items: CandidateItems) -> Optional[Dict[Any, float]]:
    """Normalize non-negative item scores into a probability distribution."""
    d = {a: max(0.0, float(p)) for a, p in items}
    s = sum(d.values())
    if s <= 0:
        return None
    return {a: p / s for a, p in d.items()}


def _sharpen_action_items(items: CandidateItems, beta: float = 2.0) -> Optional[Dict[Any, float]]:
    """Apply power sharpening and normalize the resulting distribution."""
    d = {a: max(0.0, float(p)) ** beta for a, p in items}
    s = sum(d.values())
    if s <= 0:
        return None
    return {a: p / s for a, p in d.items()}


def _softmax_from_scores(scores: List[float], temperature: float = 1.0) -> List[float]:
    """Convert augmented utilities into a numerically stable softmax distribution."""
    if not scores:
        return []
    tau = max(float(temperature), 1e-8)
    scaled = [x / tau for x in scores]
    m = max(scaled)
    exp_scores = [math.exp(x - m) for x in scaled]
    z = sum(exp_scores)
    if z <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [x / z for x in exp_scores]


def _normalize_location(loc: Optional[str]) -> Optional[str]:
    """Normalize a Diplomacy location token while preserving coast suffixes."""
    if not loc:
        return None
    s = str(loc).strip()
    if s.startswith("*"):
        s = s[1:].strip()
    return s


# Backward-compatible helper name used by older code.
_norm_loc = _normalize_location


def _base_province(loc: Optional[str]) -> Optional[str]:
    """Return the base province without coast suffixes."""
    norm = _normalize_location(loc)
    if not norm:
        return None
    return norm.split("/")[0]


def _as_state_dict(game: Any) -> Dict[str, Any]:
    """Read the current game state as a plain dictionary whenever possible."""
    try:
        st = game.get_state()
    except Exception:
        return {}
    if not isinstance(st, dict) and hasattr(st, "to_dict"):
        try:
            st = st.to_dict()
        except Exception:
            return {}
    return st if isinstance(st, dict) else {}


def _is_movement_phase(game: Any) -> bool:
    """Return whether the current phase is a movement phase."""
    try:
        return str(game.get_current_phase()).upper().endswith("M")
    except Exception:
        return False


def _action_to_orders(action: Any) -> List[str]:
    """Convert an action object into a list of order strings."""
    if isinstance(action, (list, tuple)):
        return [str(x) for x in action]
    if action is None:
        return []
    return [str(action)]


def _build_loc2power(game: Any) -> Dict[str, str]:
    """Map each currently occupied location to the occupying power."""
    st = _as_state_dict(game)
    units = st.get("units")
    if not isinstance(units, dict):
        return {}

    loc2power: Dict[str, str] = {}
    for pwr, unit_list in units.items():
        for unit in unit_list or []:
            parts = str(unit).split()
            if len(parts) >= 2:
                loc = _normalize_location(parts[1])
                if loc:
                    loc2power[loc] = str(pwr)
    return loc2power


def _owner_by_occupancy(occ: Dict[str, str], loc: Optional[str]) -> Optional[str]:
    """Resolve the owner of a currently occupied location, including coast fallback."""
    norm = _normalize_location(loc)
    if not norm:
        return None
    owner = occ.get(norm)
    if owner:
        return owner
    if "/" not in norm:
        candidates = [v for k, v in occ.items() if k.startswith(norm + "/")]
        if candidates and all(x == candidates[0] for x in candidates):
            return candidates[0]
    return None


def _owner_by_territory(
    prov_base: Optional[str],
    st: Dict[str, Any],
    occ: Dict[str, str],
    my_power: str,
) -> Optional[str]:
    """
    Resolve the power associated with a target province.

    Occupation is used first because it represents the immediate tactical
    interaction. Supply-center control and influence are then used as fallback
    territorial signals. The acting power itself is returned as None because
    relation-inconsistency scoring only needs other powers.
    """
    if not prov_base:
        return None

    hit = None
    for loc, pwr in occ.items():
        if loc == prov_base or loc.startswith(prov_base + "/"):
            if hit is None:
                hit = pwr
            elif hit != pwr:
                hit = None
                break
    if hit:
        return None if hit == my_power else hit

    centers = st.get("centers", {}) or {}
    if isinstance(centers, dict):
        for pwr, center_list in centers.items():
            for center in center_list or []:
                if str(center).split("/")[0] == prov_base:
                    pwr = str(pwr)
                    return None if pwr == my_power else pwr

    influence = st.get("influence", None)
    if isinstance(influence, dict):
        owner = None
        for pwr, territory_list in influence.items():
            for territory in territory_list or []:
                if str(territory).split("/")[0] == prov_base:
                    if owner is None:
                        owner = str(pwr)
                    elif owner != str(pwr):
                        return None
        if owner:
            return None if owner == my_power else owner

    return None


def _extract_destination_base(order_str: str) -> Optional[str]:
    """Extract the base destination province from a move-like order string."""
    toks = str(order_str).strip().split()
    if "-" in toks:
        i = toks.index("-")
        if i + 1 < len(toks):
            return _base_province(toks[i + 1])
    return None


def get_territory_parts(st: Dict[str, Any], power: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Return supply-center, unit-occupied, and historical-free territories.

    All returned locations are base provinces. The third component uses the
    optional influence field when available and removes currently occupied
    provinces so that the three sets remain interpretable as distinct signals.
    """
    units = st.get("units", {}) or {}
    centers = st.get("centers", {}) or {}
    influence = st.get("influence", None)

    sc_set = {str(x).split("/")[0] for x in centers.get(power, []) or []}

    unit_set: Set[str] = set()
    for unit in units.get(power, []) or []:
        parts = str(unit).strip().lstrip("*").split()
        if len(parts) >= 2:
            base = _base_province(parts[1])
            if base:
                unit_set.add(base)

    if not isinstance(influence, dict):
        return sc_set, unit_set, set()

    occupied_now: Set[str] = set()
    for unit_list in units.values():
        for unit in unit_list or []:
            parts = str(unit).strip().lstrip("*").split()
            if len(parts) >= 2:
                base = _base_province(parts[1])
                if base:
                    occupied_now.add(base)

    historical_owned = {str(x).split("/")[0] for x in influence.get(power, []) or []}
    past_free_set = (historical_owned - occupied_now) - sc_set - unit_set
    return sc_set, unit_set, past_free_set


def _get_last_movement_phase_snapshot(game: Any) -> Optional[Tuple[str, Dict[str, Any], Dict[str, List[str]]]]:
    """Read the latest movement-phase state and submitted orders from history."""
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

    for rec in reversed(list(hist)):
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
        if isinstance(orders, list):
            tmp: Dict[str, List[str]] = {}
            for item in orders:
                if isinstance(item, dict):
                    for k, v in item.items():
                        tmp[str(k)] = list(v) if isinstance(v, (list, tuple)) else [str(v)]
            orders = tmp
        if not isinstance(orders, dict):
            orders = {}

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


def _current_relation_sets(game: Any, my_power: str, action: Any) -> Tuple[Set[str], Set[str]]:
    """Construct current positive and negative relation target sets."""
    st = _as_state_dict(game)
    occ = _build_loc2power(game)
    helped: Set[str] = set()
    attacked: Set[str] = set()

    for order in _action_to_orders(action):
        order = str(order)

        if " S " in order or " C " in order:
            rhs = order.split(" S ", 1)[1].strip() if " S " in order else order.split(" C ", 1)[1].strip()
            rtoks = rhs.split()
            supported_loc = _normalize_location(rtoks[1]) if len(rtoks) >= 2 else None
            beneficiary = _owner_by_occupancy(occ, supported_loc)

            if beneficiary and beneficiary != my_power:
                helped.add(beneficiary)

            target = _extract_destination_base(rhs)
            target_owner = _owner_by_territory(target, st=st, occ=occ, my_power=my_power)
            if target_owner and beneficiary and target_owner != beneficiary:
                attacked.add(target_owner)
        else:
            target = _extract_destination_base(order)
            target_owner = _owner_by_territory(target, st=st, occ=occ, my_power=my_power)
            if target_owner:
                attacked.add(target_owner)

    return helped, attacked


def _historical_relation_sets(game: Any, my_power: str) -> Tuple[Set[str], Set[str], Optional[str]]:
    """Construct previous-turn helpers and attackers directed toward the acting power."""
    snap = _get_last_movement_phase_snapshot(game)
    if not snap:
        return set(), set(), None

    prev_phase, st_prev, prev_orders = snap
    units_prev = st_prev.get("units", {}) or {}
    occ_prev: Dict[str, str] = {}
    if isinstance(units_prev, dict):
        for pwr, unit_list in units_prev.items():
            for unit in unit_list or []:
                parts = str(unit).strip().lstrip("*").split()
                if len(parts) >= 2:
                    loc = _normalize_location(parts[1])
                    if loc:
                        occ_prev[loc] = str(pwr)

    sc_set, unit_set, past_free_set = get_territory_parts(st_prev, my_power)
    my_territory_prev = sc_set | unit_set | past_free_set

    last_helped_me: Set[str] = set()
    last_attacked_me: Set[str] = set()

    for pwr, order_list in (prev_orders or {}).items():
        pwr = str(pwr)
        if pwr == my_power:
            continue
        for order in order_list or []:
            order = str(order)

            if " S " in order or " C " in order:
                rhs = order.split(" S ", 1)[1].strip() if " S " in order else order.split(" C ", 1)[1].strip()
                rtoks = rhs.split()
                supported_loc = _normalize_location(rtoks[1]) if len(rtoks) >= 2 else None
                supported_owner = _owner_by_occupancy(occ_prev, supported_loc)
                if supported_owner == my_power:
                    last_helped_me.add(pwr)

            if " S " in order:
                rhs = order.split(" S ", 1)[1].strip()
                rtoks = rhs.split()
                supported_loc = _normalize_location(rtoks[1]) if len(rtoks) >= 2 else None
                supported_owner = _owner_by_occupancy(occ_prev, supported_loc)
                target = _extract_destination_base(rhs)
                if target and target in my_territory_prev and supported_owner != my_power:
                    last_attacked_me.add(pwr)
            else:
                target = _extract_destination_base(order)
                if target and target in my_territory_prev:
                    last_attacked_me.add(pwr)

    return last_helped_me, last_attacked_me, prev_phase


def compute_relation_inconsistency_score(
    game: Any,
    my_power: str,
    action: Any,
) -> Tuple[int, Dict[str, Any]]:
    """
    Compute phi: the coopetitive relation inconsistency score.

    phi combines intra-turn polarity inconsistency and inter-turn polarity
    reversal against the previous movement-phase relation graph.
    """
    if not _is_movement_phase(game):
        return 0, {
            "phi": 0,
            "intra_targets": [],
            "inter_targets": [],
            "helped_now": [],
            "attacked_now": [],
            "last_helped_me": [],
            "last_attacked_me": [],
            "prev_phase": None,
        }

    helped_now, attacked_now = _current_relation_sets(game, my_power, action)
    last_helped_me, last_attacked_me, prev_phase = _historical_relation_sets(game, my_power)

    intra = helped_now & attacked_now
    inter = (attacked_now & last_helped_me) | (helped_now & last_attacked_me)
    phi = len(intra) + len(inter)

    return phi, {
        "phi": phi,
        "intra_targets": sorted(intra),
        "inter_targets": sorted(inter),
        "helped_now": sorted(helped_now),
        "attacked_now": sorted(attacked_now),
        "last_helped_me": sorted(last_helped_me),
        "last_attacked_me": sorted(last_attacked_me),
        "prev_phase": prev_phase,
    }


def _target_location_conflicts(game: Any, my_power: str, action: Any) -> Tuple[Set[str], Dict[str, Any]]:
    """Compute destination-level sub-plan conflicts."""
    occ = _build_loc2power(game)
    helped_ally_dest2orders: Dict[str, List[str]] = {}
    my_dest2orders: Dict[str, List[str]] = {}

    for order in _action_to_orders(action):
        s = str(order).strip()
        if not s:
            continue

        if " S " in s or " C " in s:
            rhs = s.split(" S ", 1)[1].strip() if " S " in s else s.split(" C ", 1)[1].strip()
            rtoks = rhs.split()
            supported_loc = _normalize_location(rtoks[1]) if len(rtoks) >= 2 else None
            supported_owner = _owner_by_occupancy(occ, supported_loc)
            target = _extract_destination_base(rhs)
            if target and supported_owner and supported_owner != my_power:
                helped_ally_dest2orders.setdefault(target, []).append(s)
        else:
            target = _extract_destination_base(s)
            if target:
                my_dest2orders.setdefault(target, []).append(s)

    overlap = set(helped_ally_dest2orders) & set(my_dest2orders)
    multi_move = {d for d, orders in my_dest2orders.items() if len(orders) > 1}
    conflicts = overlap | multi_move

    return conflicts, {
        "target_conflicts": sorted(conflicts),
        "help_move_overlap": sorted(overlap),
        "multi_move_targets": sorted(multi_move),
        "helped_ally_dest2orders": helped_ally_dest2orders,
        "my_dest2orders": my_dest2orders,
    }


def _territorial_boundary_conflicts(game: Any, my_power: str, action: Any) -> Tuple[Set[str], Dict[str, Any]]:
    """Compute territorial-boundary conflicts from outward cooperative orders."""
    st = _as_state_dict(game)
    sc_set, unit_set, past_free_set = get_territory_parts(st, my_power)
    my_territory = sc_set | unit_set | past_free_set
    occ = _build_loc2power(game)

    conflicts: Set[str] = set()
    conflict_orders: List[str] = []

    for order in _action_to_orders(action):
        s = str(order).strip()
        if not s or (" S " not in s and " C " not in s):
            continue

        rhs = s.split(" S ", 1)[1].strip() if " S " in s else s.split(" C ", 1)[1].strip()
        rtoks = rhs.split()
        supported_loc = _normalize_location(rtoks[1]) if len(rtoks) >= 2 else None
        supported_owner = _owner_by_occupancy(occ, supported_loc)
        if not supported_owner or supported_owner == my_power:
            continue

        target = _extract_destination_base(rhs)
        if target and target in my_territory:
            conflicts.add(target)
            conflict_orders.append(s)

    return conflicts, {
        "territorial_conflicts": sorted(conflicts),
        "territorial_conflict_orders": conflict_orders,
        "sc_set": sorted(sc_set),
        "unit_set": sorted(unit_set),
        "past_free_set": sorted(past_free_set),
    }


def compute_subplan_conflict_score(
    game: Any,
    my_power: str,
    action: Any,
) -> Tuple[int, Dict[str, Any]]:
    """
    Compute psi: the sub-plan compatibility violation score.

    psi combines target-location conflicts and territorial-boundary conflicts
    among the unit-level orders contained in a candidate action.
    """
    if not _is_movement_phase(game):
        return 0, {
            "psi": 0,
            "target_conflicts": [],
            "territorial_conflicts": [],
        }

    target_conflicts, target_detail = _target_location_conflicts(game, my_power, action)
    territorial_conflicts, territorial_detail = _territorial_boundary_conflicts(game, my_power, action)
    psi = len(target_conflicts) + len(territorial_conflicts)

    detail: Dict[str, Any] = {
        "psi": psi,
        **target_detail,
        **territorial_detail,
    }
    return psi, detail


def compute_action_quality_vector(
    game: Any,
    my_power: str,
    action: Any,
    strategic_score: float,
) -> Dict[str, Any]:
    """
    Compute the candidate-level quality vector q=(u, phi, psi).

    strategic_score is the value proxy supplied by the underlying policy/search
    distribution in this implementation.
    """
    phi, relation_detail = compute_relation_inconsistency_score(game, my_power, action)
    psi, subplan_detail = compute_subplan_conflict_score(game, my_power, action)
    return {
        "action": tuple(_action_to_orders(action)),
        "u": float(strategic_score),
        "phi": int(phi),
        "psi": int(psi),
        "relation_detail": relation_detail,
        "subplan_detail": subplan_detail,
    }


def regularize_action_set_by_structure(
    game: Any,
    my_power: str,
    items: CandidateItems,
    *,
    temperature: float = 1.0,
) -> Tuple[CandidateItems, List[Dict[str, Any]]]:
    """
    Apply CR^2 structural regularization to a candidate action set.

    Input:
      - items: List[(action, strategic_score)]

    Output:
      - regularized_items: List[(action, probability)]
      - structural_metrics: candidate-level q, augmented utility, and details

    No candidate is removed. Structural issues reduce the augmented utility and
    therefore reduce the candidate's sampling probability.
    """
    if not items:
        return [], []

    qualities = [compute_action_quality_vector(game, my_power, action, score) for action, score in items]
    strategic_values = [q["u"] for q in qualities]
    value_range = max(strategic_values) - min(strategic_values)
    if value_range <= 1e-12:
        value_range = max(max(abs(x) for x in strategic_values), 1.0)

    augmented_utilities: List[float] = []
    for q in qualities:
        penalty = value_range * (q["phi"] + q["psi"])
        augmented = q["u"] - penalty
        q["delta_u"] = value_range
        q["structural_penalty"] = penalty
        q["augmented_u"] = augmented
        augmented_utilities.append(augmented)

    probs = _softmax_from_scores(augmented_utilities, temperature=temperature)
    regularized_items: CandidateItems = []
    structural_metrics: List[Dict[str, Any]] = []

    for (action, _), prob, q in zip(items, probs, qualities):
        normalized_action = tuple(_action_to_orders(action))
        regularized_items.append((normalized_action, prob))
        structural_metrics.append({
            **q,
            "regularized_prob": prob,
        })

    regularized_items.sort(key=lambda kv: kv[1], reverse=True)
    structural_metrics.sort(key=lambda x: x["regularized_prob"], reverse=True)
    return regularized_items, structural_metrics


def check_c1_intra_turn_consistency(
    game: Any,
    my_power: str,
    action: Any,
) -> Tuple[bool, str]:
    """Check intra-turn polarity inconsistency."""
    if not _is_movement_phase(game):
        return True, ""
    helped, attacked = _current_relation_sets(game, my_power, action)
    overlap = helped & attacked
    if overlap:
        return False, f"C1_INTRA_TURN: overlap={sorted(overlap)} helped={sorted(helped)} attacked={sorted(attacked)}"
    return True, ""


def check_c2_inter_turn_consistency(
    game: Any,
    my_power: str,
    action: Any,
) -> Tuple[bool, str]:
    """Check inter-turn polarity reversal against previous relation signals."""
    if not _is_movement_phase(game):
        return True, ""

    helped_now, attacked_now = _current_relation_sets(game, my_power, action)
    last_helped_me, last_attacked_me, prev_phase = _historical_relation_sets(game, my_power)

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
    """Check target-location sub-plan conflicts."""
    if not _is_movement_phase(game):
        return True, ""

    conflicts, detail = _target_location_conflicts(game, my_power, action)
    if not conflicts:
        return True, ""

    parts: List[str] = []
    help_overlap = detail.get("help_move_overlap", [])
    multi_move = detail.get("multi_move_targets", [])

    if help_overlap:
        overlap_detail = []
        for target in help_overlap:
            help_orders = detail["helped_ally_dest2orders"].get(target, [])
            move_orders = detail["my_dest2orders"].get(target, [])
            overlap_detail.append(f"{target}: help={help_orders} | move={move_orders}")
        parts.append("help_move_overlap=" + "; ".join(overlap_detail))

    if multi_move:
        multi_detail = []
        for target in multi_move:
            multi_detail.append(f"{target}: moves={detail['my_dest2orders'].get(target, [])}")
        parts.append("multi_move=" + "; ".join(multi_detail))

    return False, "C3_DEST_CONFLICT: " + " | ".join(parts)


def check_c4_self_defense_consistency(
    game: Any,
    my_power: str,
    action: Any,
) -> Tuple[bool, str]:
    """Check territorial-boundary conflicts caused by outward cooperative orders."""
    if not _is_movement_phase(game):
        return True, ""

    conflicts, detail = _territorial_boundary_conflicts(game, my_power, action)
    if not conflicts:
        return True, ""

    return False, (
        "C4_TERRITORIAL_BOUNDARY: "
        f"targets={detail.get('territorial_conflicts', [])} "
        f"orders={detail.get('territorial_conflict_orders', [])}"
    )
