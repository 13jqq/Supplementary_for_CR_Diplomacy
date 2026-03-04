# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import itertools
import math
from collections import defaultdict

POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
@dataclass
class CoalitionGraph:
    """
    Me-centric coalition graph.
    Only maintains weights for (me, j) where j != me.
    """
    V: List[str]
    w: Dict[Tuple[str, str], float]          # keys always (me, j)
    E: List[Tuple[str, str]]                 # top-k edges from w
    me: str
    chosen_allies: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)



@dataclass
class CoalitionSolverConfig:
    tau: int = 6
    top_k: int = 4
    min_coalition_size: int = 1
    max_coalition_size: int = 2
    n_rollouts: int = 16
    support_bonus: float = 1.5
    prior_affinity: Dict[Tuple[str, str], float] = field(default_factory=dict)
    eps_me: float = 0.0
    eps_ally: float = 0.0

# ---------------------------
# Default priors (as you described)
# ---------------------------
_PREFERRED = {
    "ENGLAND": (["GERMANY", "FRANCE"], ["RUSSIA"]),
    "FRANCE":  (["GERMANY", "ENGLAND"], []),
    "GERMANY": (["AUSTRIA", "FRANCE", "ENGLAND"], []),
    "ITALY":   (["AUSTRIA", "RUSSIA"], ["TURKEY"]),
    "AUSTRIA": (["GERMANY", "ITALY", "RUSSIA"], ["TURKEY"]),
    "RUSSIA":  (["TURKEY", "AUSTRIA"], ["ENGLAND"]),
    "TURKEY":  (["RUSSIA"], ["AUSTRIA", "ITALY"]),
}

# ---------------------------
# Default priors（返回的是 T 分数，不是 weight）
# ---------------------------

_PRIOR_T_PREF = 5.0
_PRIOR_T_NEUT = 2.5
_PRIOR_T_BAD  = 0.0

def build_default_prior_affinity() -> Dict[Tuple[str, str], float]:
    """
    返回“初始联盟倾向分数 T”（不是边权重 w）：

    keys stored as (me, other).
    """
    pri_T: Dict[Tuple[str, str], float] = {}
    for me in POWERS:
        pref, bad = _PREFERRED.get(me, ([], []))
        for other in POWERS:
            if other == me:
                continue
            if other in pref:
                pri_T[(me, other)] = _PRIOR_T_PREF
            elif other in bad:
                pri_T[(me, other)] = _PRIOR_T_BAD
            else:
                pri_T[(me, other)] = _PRIOR_T_NEUT
    return pri_T



def T_to_weight(T: float, T_max: float = 5.0) -> float:
    """
    线性映射：T ∈ [0, T_max] -> w ∈ [0, 1]
    你设的：T=5,2.5,0 会严格变成 1,0.5,0
    """
    if T_max <= 0:
        return 0.0
    w = T / T_max
    # 截断到 [0,1]
    if w < 0.0: 
        return 0.0
    if w > 1.0:
        return 1.0
    return w

DELTA = {
    "SUPPORT_USED": +3.0,
    "SUPPORT_UNUSED": +1.5,
    "CONVOY_USED": +3.0,
    "CONVOY_UNUSED": +1.5,
    "ATTACK_TERRITORY": -3.0,
    "SUPPORT_ENEMY_INTO_MY_TERRITORY": -1.5,
}



class CoalitionSolver:
    def __init__(self, cfg: CoalitionSolverConfig):
        self.cfg = cfg
        # if empty, fill with default prior
        if not self.cfg.prior_affinity:
            self.cfg.prior_affinity = build_default_prior_affinity()
        # 存每个 (me, j) 的“联盟倾向分数 T”（跨 phase 记忆）
        self.T: Dict[Tuple[str, str], float] = {}
    
    
    # ============================
    # NEW: T-scoring + evidence
    # ============================
    def _update_T_from_history(self, me: str, H_t: Any) -> Tuple[
        Dict[str, float],
        Dict[str, float],
        Dict[str, List[Dict[str, Any]]],
        Dict[str, Dict[str, float]],
    ]:
        """
        用 H_t 里的 support_used/support_unused/attack_last 进行 T 更新。
        返回：
          T_prev: power->T_before
          T_new : power->T_after
          evidence: power->[{order, delta, tag}, ...]
        """
        # 0) ensure T initialized from prior (T-score)
        T_prev: Dict[str, float] = {}
        T_new: Dict[str, float] = {}
        evidence: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        T_calc: Dict[str, Dict[str, float]] = {}


        for j in POWERS:
            if j == me:
                continue
            key = (me, j)
            if key not in self.T:
                self.T[key] = float(self.cfg.prior_affinity.get(key, _PRIOR_T_NEUT))
            T_prev[j] = float(self.T[key])

        # 1) read last-round signals
        support_used = (H_t or {}).get("support_used", {}) or {}
        support_unused = (H_t or {}).get("support_unused", {}) or {}
        attack_last = (H_t or {}).get("attack_last", {}) or {}

        def _has_token(order: str, tok: str) -> bool:
            # robust "contains token" check
            s = f" {order.strip()} "
            return f" {tok} " in s

        # 2) apply deltas per opponent
        for j in POWERS:
            if j == me:
                continue
            key = (me, j)
            delta_total = 0.0

            # --- support used ---
            for od in (support_used.get(j, []) or []):
                tag = "CONVOY_USED" if _has_token(od, "C") else "SUPPORT_USED"
                d = float(DELTA.get(tag, DELTA["SUPPORT_USED"]))
                evidence[j].append({"order": od, "delta": d, "tag": tag})
                delta_total += d

            # --- support unused ---
            for od in (support_unused.get(j, []) or []):
                tag = "CONVOY_UNUSED" if _has_token(od, "C") else "SUPPORT_UNUSED"
                d = float(DELTA.get(tag, DELTA["SUPPORT_UNUSED"]))
                evidence[j].append({"order": od, "delta": d, "tag": tag})
                delta_total += d

            # --- attacks (distinguish direct move vs support/convoy into my terr) ---
            for od in (attack_last.get(j, []) or []):
                if _has_token(od, "S") or _has_token(od, "C"):
                    tag = "SUPPORT_ENEMY_INTO_MY_TERRITORY"
                else:
                    tag = "ATTACK_TERRITORY"
                d = float(DELTA.get(tag, 0.0))
                evidence[j].append({"order": od, "delta": d, "tag": tag})
                delta_total += d

            T_before = float(self.T[key])
            T_raw = T_before + float(delta_total)
            T_clipped = max(0.0, min(T_raw, 10.0))  # ✅ 限制在[0, 10]

            if delta_total != 0.0:
                self.T[key] = T_clipped

            T_new[j] = float(self.T[key])

            # ✅ 记录完整计算过程（用于日志输出公式）
            T_calc[j] = {
                "T_before": T_before,
                "delta_total": float(delta_total),
                "T_raw": T_raw,
                "T_clipped": T_clipped,
            }


        return T_prev, T_new, dict(evidence), T_calc


    def solve_coalition(
        self,
        game: Any,       # pydipcc.Game
        me: str,
        pi_me: Any,      # placeholder for now
        pi_bp: Any,      # placeholder for now
        H_t: Any,        # placeholder for now
        step: int,       # movement step
    ) -> CoalitionGraph:
        phase = str(game.get_current_phase())

        # 1) update T from last-round evidence
        T_prev, T_new, evidence, T_calc = self._update_T_from_history(me, H_t)


        # 2) map T -> weight w (only (me,j))
        w: Dict[Tuple[str, str], float] = {}
        for j in POWERS:
            if j == me:
                continue
            w[(me, j)] = float(T_to_weight(self.T[(me, j)], T_max=_PRIOR_T_PREF))


        # 3) sparsify edges: keep top_k
        items = sorted(w.items(), key=lambda kv: kv[1], reverse=True)
        top_k = max(1, min(self.cfg.top_k, len(items)))
        E = [k for k, _ in items[:top_k]]  # list of (me, j)

        # 4) choose allies S* (skeleton: take top from candidates)
        candidates = [j for (_, j) in E]
        k_min = max(1, self.cfg.min_coalition_size)
        k_max = max(k_min, self.cfg.max_coalition_size)
        k = min(k_max, len(candidates))
        chosen_allies = candidates[:k]
        if len(chosen_allies) < k_min and candidates:
            chosen_allies = candidates[:k_min]

        meta = {
            "phase": phase,
            "step": step,
            "candidates": candidates,
            "candidate_EU": {j: w[(me, j)] for j in candidates},

            # ✅ for logging/verification
            "T_prev": T_prev,
            "T_new": T_new,
            "evidence": evidence,

            "rollouts_used": 0,
            "eps_me": self.cfg.eps_me,
            "eps_ally": self.cfg.eps_ally,
        }

        return CoalitionGraph(
            V=list(POWERS),
            w=w,
            E=E,
            me=me,
            chosen_allies=chosen_allies,
            meta=meta,
        )