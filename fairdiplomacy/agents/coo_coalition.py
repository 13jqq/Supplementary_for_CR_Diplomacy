# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import itertools
import math


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
class CoalitionGraph:
    V: List[str]
    w: Dict[Tuple[str, str], float]          # only (me, j)
    E: List[Tuple[str, str]]                 # top_k edges from w
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

def build_default_prior_affinity() -> Dict[Tuple[str, str], float]:
    """
    All priors mentioned:
      preferred -> 0.5
      disliked  -> 0.0
      not mentioned -> 0.25
    keys stored as (me, other).
    """
    pri: Dict[Tuple[str, str], float] = {}
    for me in POWERS:
        pref, bad = _PREFERRED.get(me, ([], []))
        for other in POWERS:
            if other == me:
                continue
            if other in pref:
                pri[(me, other)] = 0.5
            elif other in bad:
                pri[(me, other)] = 0.0
            else:
                pri[(me, other)] = 0.25
    return pri

class CoalitionSolver:
    def __init__(self, cfg: CoalitionSolverConfig):
        self.cfg = cfg
        # if empty, fill with default prior
        if not self.cfg.prior_affinity:
            self.cfg.prior_affinity = build_default_prior_affinity()

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

        # 1) compute weights w[(me,j)] using prior only (skeleton)
        w: Dict[Tuple[str, str], float] = {}
        for j in POWERS:
            if j == me:
                continue
            w[(me, j)] = float(self.cfg.prior_affinity.get((me, j), 0.25))

        # 2) sparsify edges: keep top_k
        items = sorted(w.items(), key=lambda kv: kv[1], reverse=True)
        top_k = max(1, min(self.cfg.top_k, len(items)))
        E = [k for k, _ in items[:top_k]]  # list of (me, j)

        # 3) choose allies S* (skeleton: take top from candidates)
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
            # placeholders for later
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