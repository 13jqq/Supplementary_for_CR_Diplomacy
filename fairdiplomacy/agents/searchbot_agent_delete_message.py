#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict

import math
from typing import Callable, DefaultDict, Dict, List, Set, Tuple, Optional, Any
import collections
import copy
import functools
import itertools
import json
import logging
import random
import tabulate
import time
from termcolor import colored

import numpy as np
from fairdiplomacy.utils.agent_interruption import raise_if_should_stop
from fairdiplomacy.utils.typedefs import get_last_message
from parlai_diplomacy.utils.game2seq.format_helpers.misc import INF_SLEEP_TIME
import torch

from conf import agents_cfgs
from fairdiplomacy.pydipcc import Game, CFRStats
# fairdiplomacy/pydipcc.cpython-37m-x86_64-linux-gnu.so
# .so 的初始化函数里用 pybind11 把 dipcc::Game 绑定成 Python 的 Game;dipcc::CFRStats 绑定成 Python 的 CFRStats
from fairdiplomacy.agents.base_agent import AgentState
from fairdiplomacy.agents.base_search_agent import (
    BaseSearchAgent,
    SearchResult,
    make_set_orders_dicts,
    sample_orders_from_policy,
)
from fairdiplomacy.agents.bilateral_stats import BilateralStats
from fairdiplomacy.pseudo_orders import PseudoOrders
from fairdiplomacy.utils.temp_redefine import temp_redefine
from parlai_diplomacy.utils.misc import last_dict_key
from parlai.utils.logging import _is_interactive

# fairdiplomacy.action_generation and fairdiplomacy.action_exploration
# both circularly refer fairdiplomacy.agents, so we import those modules whole
# instead of from "fairdiplomacy.action_generation import blah".
# That way, we break the circular initialization issue by not requiring the symbols
# *within* those modules to exist at import time, since they very well might not exist
# if we were only halfway through importing those when we began importing this file.
import fairdiplomacy.action_generation
import fairdiplomacy.action_exploration
from fairdiplomacy.agents.base_strategy_model_rollouts import (
    BaseStrategyModelRollouts,
    RolloutResultsCache,
)
from fairdiplomacy.agents.base_strategy_model_wrapper import BaseStrategyModelWrapper
from fairdiplomacy.agents.plausible_order_sampling import (
    PlausibleOrderSampler,
    cutoff_policy,
    renormalize_policy,
)
from fairdiplomacy.agents.br_corr_bilateral_search import (
    BRCorrBilateralSearchResult,
    compute_payoff_matrix_for_all_opponents,
    extract_bp_policy_for_powers,
)
from fairdiplomacy.models.consts import POWER2IDX, POWERS
from fairdiplomacy.utils.game import game_from_two_party_view, get_last_message_from, next_M_phase
from fairdiplomacy.utils.parse_device import device_id_to_str
from fairdiplomacy.utils.sampling import sample_p_dict, argmax_p_dict
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.order_idxs import is_action_valid
from fairdiplomacy.utils.base_strategy_model_multi_gpu_wrappers import (
    MultiProcessBaseStrategyModelExecutor,
)
from fairdiplomacy.viz.meta_annotations import api as meta_annotations
from fairdiplomacy.typedefs import (
    Action,
    BilateralConditionalValueTable,
    JointAction,
    MessageDict,
    MessageHeuristicResult,
    ConditionalValueTable,
    Phase,
    PlausibleOrders,
    PlayerRating,
    Policy,
    Power,
    PowerPolicies,
    Timestamp,
)
from parlai_diplomacy.wrappers.dialogue import TOKEN_DETAILS_TAG
from parlai_diplomacy.wrappers.factory import load_order_wrapper
from parlai_diplomacy.wrappers.orders import ParlAIPlausiblePseudoOrdersWrapper
from parlai_diplomacy.wrappers.base_wrapper import RolloutType

from fairdiplomacy.agents.parlai_message_handler import (
    ParlaiMessageHandler,
    ParlaiMessagePseudoOrdersCache,
    SleepSixTimesCache,
    pseudoorders_initiate_sleep_heuristics_should_trigger,
    joint_action_contains_xpower_support_or_convoy,
)


ActionDict = Dict[Tuple[Power, Action], float]

class CFRResult(SearchResult):
    # CFRData 是 Python 这边对 CFRStats （C++ 的 cfrstats.cc（CFRStats 类））的一层包装
    # run_search(...) 在循环过程中不断调用 dipcc 里的 CFRStats 去更新遗憾和策略，
    # 等所有 CFR 迭代跑完，再把从 CFRStats 里读出的 avg_policy / cur_iter_policy / avg_utility 等封装到 CFRResult 里，并返回给上层。
    def __init__(
        self,
        bp_policies: PowerPolicies,
        avg_policies: PowerPolicies,
        final_policies: PowerPolicies,
        cfr_data: Optional["CFRData"],
        use_final_iter: bool,
        bilateral_stats: Optional[BilateralStats] = None,
    ):
        self.bp_policies = bp_policies # bp_policies：蓝图策略（plausible orders + 概率）
        self.avg_policies = avg_policies # avg_policies：CFR 平均策略（通常当作最终策略用）
        self.final_policies = final_policies # final_policies：最后一轮策略（cur_iter_policy）
        self.cfr_data = cfr_data  # type: ignore # cfr_data：指向 CFRData，从而可以继续访问各种平均 utility / regret / action prob 等内部信息
        self.use_final_iter = use_final_iter
        self.bilateral_stats = bilateral_stats

    def get_agent_policy(self) -> PowerPolicies:
        return self.avg_policies

    def get_population_policy(self) -> PowerPolicies:
        return self.avg_policies

    def get_bp_policy(self) -> PowerPolicies:
        return self.bp_policies

    def sample_action(self, power) -> Action:
        policies = self.final_policies if self.use_final_iter else self.avg_policies
        return sample_p_dict(policies[power])

    def avg_utility(self, pwr: Power) -> float:
        return self.cfr_data.avg_utility(pwr) if self.cfr_data is not None else 0

    def avg_action_utility(self, pwr: Power, a: Action) -> float:
        return self.cfr_data.avg_action_utility(pwr, a) if self.cfr_data is not None else 0

    def get_bilateral_stats(self) -> BilateralStats:
        assert self.bilateral_stats is not None
        return self.bilateral_stats

    def is_early_exit(self) -> bool:
        return self.cfr_data is None

# 没有可用的命令或者只有一个命令时返回
def _early_quit_cfr_result(power: Power, *, action: Action = tuple()) -> CFRResult:
    policies = {power: {action: 1.0}}
    return CFRResult(
        bp_policies=policies,
        avg_policies=policies,
        final_policies=policies,
        cfr_data=None,
        use_final_iter=False,
    )

#按概率从大到小排序，然后再做成一个有序的 dict。
def sorted_policy(plausible_orders: List[Action], probs: List[float]) -> Policy:
    return dict(sorted(zip(plausible_orders, probs), key=lambda ac_p: -ac_p[1]))


class CFRData:
    def __init__(
        self,
        bp_policy: PowerPolicies,
        use_optimistic_cfr: bool,
        qre: Optional[agents_cfgs.SearchBotAgent.QRE] = None,
        agent_power=None,
        scale_lambdas_by_power: Optional[Dict[Power, float]] = None,
    ):
        # Make sure that all powers have some actions. This guarantees that
        # we run utility computation for every power and so state values will
        # be computed correctly. In theory, only alive powers should have
        # non-zero utility, ano so we only need to augment alive powers without
        # orders.  But in practice as the state values are computed by a value
        # network some dead power may have non-zero utilities.
        # Note, that the plausible order sampler by default adds empty policies
        # for all powers.
        # 确保所有权力都有行动。这保证了我们对每个权力都进行效用计算，从而正确计算状态值。
        # 理论上，只有存活的权力才应该具有非零效用，因此我们只需要增强没有指令的存活权力。
        # 但实际上，由于状态值是由一个值网络计算的，一些失效的权力可能具有非零效用。
        # 请注意，默认情况下，合理的指令采样器会为所有权力添加空策略。

        # 输入： bp_policy蓝图策略 prob 可以是 0，但键必须存在（即每个 Power 至少有一个动作）
        # use_optimistic_cfr 控制 CFRStats 内部是否用 optimistic/hedge 风格的加权。
        # qre 若非空，则启用 QRE（Quantal Response Equilibrium） 修正：把蓝图概率当成熵正则/先验的一部分；支持 agent 专属的 λ/entropy 因子。（在这个算法中没用）（后续可以直接改为True试一试）
        # 剩下两个也没用

        for power in bp_policy:
            assert bp_policy[
                power
            ], f"Power {power} doesn't have policy. Add an empty action policy."

        self.use_optimistic_cfr = use_optimistic_cfr
        use_linear_weighting = True
        # print("qre", qre)
        # input()
        # 无QRE
        if qre:
            input("qre")
            # 应该是用不到QRE的我先删了
        else:
            use_qre = False
            qre_target_blueprint = False
            qre_eta = 0.0
            qre_lambda = 0.0
            qre_entropy_factor = 1.0
            power_qre_lambda = {p: qre_lambda for p in POWERS}
            power_qre_entropy_factor = {p: qre_entropy_factor for p in POWERS}

        # print("scale_lambdas_by_power", scale_lambdas_by_power)
        # input()
        # 这里没有scale_lambdas_by_power
        if scale_lambdas_by_power is not None:
            for power in scale_lambdas_by_power:
                power_qre_lambda[power] *= scale_lambdas_by_power[power]

        # print("power_qre_lambda", power_qre_lambda)
        # input()
        # power_qre_lambda {'AUSTRIA': 0.0, 'ENGLAND': 0.0, 'FRANCE': 0.0, 'GERMANY': 0.0, 'ITALY': 0.0, 'RUSSIA': 0.0, 'TURKEY': 0.0}
        self.power_qre_lambda = power_qre_lambda
        # 这个self.power_qre_lambda也一直是0
        self.power_plausible_orders: PlausibleOrders = {p: sorted(v) for p, v in bp_policy.items()}
        # 注意这里对动作本身（Action）排序，不是按概率。这样保证每个 Power 的动作列表是固定顺序，方便用“下标”与 CFRStats 内部张量对齐。
        # 总之 是按照所有命令的字符大小排了一个先后顺序
        power_plausible_action_probs = {
            p: [bp_policy[p][a] for a in self.power_plausible_orders[p]] for p in POWERS
        }
        # print("power_plausible_action_probs", power_plausible_action_probs)
        # input()

        # 这个初始化将Python的字典格式转换为C++的向量格式。
        self.stats = CFRStats(
            use_linear_weighting,
            use_optimistic_cfr,
            use_qre,
            qre_target_blueprint,
            qre_eta,
            power_qre_lambda,
            power_qre_entropy_factor,
            power_plausible_action_probs,
        )

    def cur_iter_strategy(self, pwr: Power) -> List[float]:
        # 从C++获取当前轮的概率列表
        return self.stats.cur_iter_strategy(pwr)

    def cur_iter_policy(self, pwr: Power) -> Policy:
        #  转换为Python字典格式
        return sorted_policy(self.power_plausible_orders[pwr], self.cur_iter_strategy(pwr))

    def avg_strategy(self, pwr: Power) -> List[float]:
        # 获取所有CFR迭代的平均策略（这是CFR收敛的关键）
        return self.stats.avg_strategy(pwr)

    def avg_policy(self, pwr: Power) -> Policy:
        # 转换为字典格式
        return sorted_policy(self.power_plausible_orders[pwr], self.avg_strategy(pwr))

    def avg_utility(self, pwr: Power) -> float:
        # 该国家的平均得分
        return self.stats.avg_utility(pwr)

    def avg_action_utilities(self, pwr: Power) -> List[float]:
        # 每个行动的平均得分列表
        return self.stats.avg_action_utilities(pwr)

    def avg_action_utility(self, pwr: Power, a: Action) -> float:
        # 特定行动的平均得分
        return self.stats.avg_action_utility(pwr, self.power_plausible_orders[pwr].index(a))

    def avg_action_regret(self, pwr: Power, a: Action) -> float:
        # 获取特定行动的平均遗憾值
        return self.stats.avg_action_regret(pwr, self.power_plausible_orders[pwr].index(a))

    def avg_action_prob(self, pwr: Power, a: Action) -> float:
        #  特定行动在平均策略中的概率
        return self.stats.avg_action_prob(pwr, self.power_plausible_orders[pwr].index(a))

    def cur_iter_action_prob(self, pwr: Power, a: Action) -> float:
        # 特定行动在当前策略中的概率
        return self.stats.cur_iter_action_prob(pwr, self.power_plausible_orders[pwr].index(a))

    def bp_strategy(self, pwr: Power, temperature=1.0) -> List[float]:
        return self.stats.bp_strategy(pwr, temperature)

    def bp_policy(self, pwr: Power, temperature=1.0) -> Policy:
        return sorted_policy(self.power_plausible_orders[pwr], self.bp_strategy(pwr, temperature))

    # CFR中的那个更新(通过C++)
    def update(
        self,
        pwr: Power,
        actions: List[Action],
        state_utility: float,
        action_utilities: List[float],
        which_strategy_to_accumulate: int,
        cfr_iter: int,
    ) -> None:
        self.stats.update(
            pwr, state_utility, action_utilities, which_strategy_to_accumulate, cfr_iter
        )


PhaseKey = Tuple[Phase, Phase]  # (dialogue_phase, rollout_phase)


class SearchBotAgentState(AgentState):
    """Cached state for a particular agent power.“某个 power 的 SearchBot，在这一盘游戏里目前记住了什么”"""

    def __init__(self, agent_power):
        self.last_ts: Timestamp = Timestamp.from_seconds(-1) #上一次更新 state 时的“最后一条消息的时间戳”。用于之后判断当前 game.messages 里有没有新消息。
        self.agent_power = agent_power # 这个 state 绑定的是哪个国家
        # hide these because we have to check that we're on the same phase!
        self._last_search_result: Dict[PhaseKey, Optional[SearchResult]] = {} #按 phase 缓存上一次的搜索结果。
        # key 是 PhaseKey = (dialogue_phase对话阶段 ,rollout_phase推演阶段)，例如：
        # 这样就不会把“同一个 movement phase，但对话阶段不同”的搜索结果混用。
        self._last_pseudo_orders: Dict[PhaseKey, Optional[JointAction]] = {}
        #同样按 phase 缓存 pseudo orders（主要给“message/伪指令搜索”用）
        self._value_table_cache: DefaultDict[
            PhaseKey, DefaultDict[Power, BilateralConditionalValueTable]
        ] = defaultdict(functools.partial(defaultdict, dict))#按 phase & power 存条件价值表，给双边搜索 / message 价值评估用。SearchBot 里有 BilateralStats、compute_payoff_matrix_for_all_opponents 之类，这里相当于它们的缓存挂载点。

        self.pseudo_orders_cache = ParlaiMessagePseudoOrdersCache() #专门给 ParlaiMessageHandler 用的 pseudo-orders 缓存；你可以理解为“语言模型根据对话生成的伪指令的 memo”。

    def _get_phase_key(self, game: Game) -> PhaseKey:
        #phase 的唯一标识
        return (
            game.get_metadata("last_dialogue_phase") or game.current_short_phase,
            game.current_short_phase,
        )

    def update(
        self,
        game: Game,
        agent_power: Power,
        search_result: Optional[SearchResult],
        pseudo_orders: Optional[JointAction],
    ) -> None:
        assert self.agent_power == agent_power
        self.last_ts = (
            last_dict_key(game.messages) if game.messages else Timestamp.from_seconds(-1)
        ) #更新消息时间戳
        self.agent_power = agent_power
        if search_result and search_result.is_early_exit():
            # don't include early-exit search results, because they
            # don't have the actual search policies for anyone
            #它里面没有真正的 CFR 策略，所以不存这个策略
            search_result = None

        phase_key = self._get_phase_key(game)
        self._last_search_result[phase_key] = search_result
        self._last_pseudo_orders[phase_key] = pseudo_orders

    def get_last_search_result(self, game: Game) -> Optional[SearchResult]:
        # 获取上一次的搜索结果
        # 使用方法：如果 自上次搜索以来没有新消息，就直接复用上一次的 SearchResult
        phase_key = self._get_phase_key(game)
        return self._last_search_result.get(phase_key, None)

    def get_last_pseudo_orders(self, game: Game) -> Optional[JointAction]:
        phase_key = self._get_phase_key(game)
        return self._last_pseudo_orders.get(phase_key, None)

    def get_new_messages(self, game: Game):
        """Return all new messages in the game since this state"""
        # 获取自上次更新 state 以来所有的“新消息”
        return [m for m in game.messages.values() if m["time_sent"] > self.last_ts]


class SearchBotAgent(BaseSearchAgent):
    """One-ply cfr with base_strategy_model-policy rollouts"""

    #在 __init__ 里，SearchBot 通过 cfg 把训练好的权重装进几个包装器

    def __init__(self, cfg: agents_cfgs.SearchBotAgent, *, skip_base_strategy_model_cache=False):
        super().__init__(cfg)
        base_strategy_model_wrapper_kwargs = dict(
            device=device_id_to_str(cfg.device),
            max_batch_size=cfg.max_batch_size,
            half_precision=cfg.half_precision,
            skip_base_strategy_model_cache=skip_base_strategy_model_cache,
        )
        self.base_strategy_model = BaseStrategyModelWrapper(
            cfg.rollout_model_path or cfg.model_path,
            value_model_path=cfg.value_model_path,
            force_disable_all_power=True,
            **base_strategy_model_wrapper_kwargs,
        )
        # 由 cfg.model_path / rollout_model_path + value_model_path 指定
        # 这是“策略/价值一体”的基础模型接口：对给定状态 + 一组联合作令，它能跑前向得到各阵营的价值评估和/或给出候选动作/概率。
        
        # input()
        # print("base_strategy_model", self.base_strategy_model)
        # print(cfg.model_path)
        # print(cfg.value_model_path)
          
        self.cfg = cfg
        self.has_press = cfg.dialogue is not None
        self.set_player_rating = cfg.set_player_rating
        self.player_rating = cfg.player_rating
        if self.set_player_rating:
            if cfg.cache_rollout_results:
                logging.warning("Undefined behaviour if searchbot.cache_rollout_results is set")
            assert (
                self.player_rating is not None and 0 <= self.player_rating <= 1.0
            ), "Player rating needs to be a float between 0 and 1.0"
            logging.info(f"Setting player rating to {self.player_rating}")
        else:
            if self.player_rating is not None:
                logging.warning(
                    "searchbot.player_rating is set but searchbot.set_player_rating is not set"
                )
            self.player_rating = None

        self.base_strategy_model_rollouts = BaseStrategyModelRollouts(
            self.base_strategy_model,
            cfg=cfg.rollouts_cfg,
            has_press=self.has_press,
            set_player_ratings=self.set_player_rating,
        )
        # 一个专门用于执行 rollout（推演）的组件，它的主要任务是：

        # 从当前游戏状态开始，使用基础策略模型来模拟游戏的后续进程
        # 评估不同行动选择的长期价值
        # 为 CFR 搜索算法提供价值估计
        self.bilateral_cfg = cfg.bilateral_dialogue
        assert cfg.n_rollouts >= 0, "Set searchbot.n_rollouts"

        self.qre = cfg.qre
        if self.qre is not None:
            logging.info(
                f"Performing qre regret minimization with eta={self.qre.eta} "
                f"and lambda={self.qre.qre_lambda} with target pi={self.qre.target_pi}"
            )
            if self.qre.qre_lambda == 0.0:
                logging.info("Using lambda 0.0 simplifies qre to regular hedge")

        self.n_rollouts = cfg.n_rollouts
        self.cache_rollout_results = cfg.cache_rollout_results
        self.precompute_cache = cfg.precompute_cache
        self.enable_compute_nash_conv = cfg.enable_compute_nash_conv
        self.n_plausible_orders = cfg.plausible_orders_cfg.n_plausible_orders
        self.use_optimistic_cfr = cfg.use_optimistic_cfr
        self.use_final_iter = cfg.use_final_iter
        self.bp_iters = cfg.bp_iters
        self.bp_prob = cfg.bp_prob
        self.loser_bp_iter = cfg.loser_bp_iter
        self.loser_bp_value = cfg.loser_bp_value
        self.reset_seed_on_rollout = cfg.reset_seed_on_rollout
        self.max_seconds = cfg.max_seconds
        self.br_corr_bilateral_search_cfg = cfg.br_corr_bilateral_search
        self.message_search_cfg = cfg.message_search

        self.all_power_base_strategy_model_executor = None
        if self.br_corr_bilateral_search_cfg is not None:
            assert (
                self.base_strategy_model.model.is_all_powers()
            ), "br_corr_bilateral_search requires an all-powers base_strategy_model model."
            allpower_wrapper_kwargs = {
                **base_strategy_model_wrapper_kwargs,
                "model_path": self.base_strategy_model.model_path,
                "value_model_path": cfg.value_model_path,
            }
            allpower_rollouts_kwargs = {
                "cfg": cfg.rollouts_cfg,
                "has_press": self.has_press,
                "set_player_ratings": self.set_player_rating,
            }
            # if we allow multi gpu for plausible orders,
            # then we also allow multi gpu for the base_strategy_model to avoid adding redundant flags
            self.all_power_base_strategy_model_executor = MultiProcessBaseStrategyModelExecutor(
                allow_multi_gpu=cfg.plausible_orders_cfg.allow_multi_gpu,
                base_strategy_model_wrapper_kwargs=allpower_wrapper_kwargs,
                base_strategy_model_rollouts_kwargs=allpower_rollouts_kwargs,
            )

        #加载parlai语言模型，searchbot没有，cicero有
        if cfg.parlai_model_orders.model_path:
            if cfg.rescoring_blueprint_model_path is not None:
                raise RuntimeError(
                    "You probably don't want to rescore parlai policy with a base_strategy_model BP"
                )

            logging.info("Setting up parlai orders model...")
            self.parlai_model_orders = load_order_wrapper(cfg.parlai_model_orders)
        else:
            self.parlai_model_orders = None

        self.cfr_messages = cfg.cfr_messages

        if cfg.dialogue is not None:
            self.message_handler = ParlaiMessageHandler(
                cfg.dialogue,
                model_orders=self.parlai_model_orders,
                base_strategy_model=self.base_strategy_model,
            )
        else:
            self.message_handler = None
            assert not self.cfr_messages

        if cfg.rollout_model_path and cfg.model_path != cfg.rollout_model_path:
            self.proposal_base_strategy_model = BaseStrategyModelWrapper(
                cfg.model_path, **base_strategy_model_wrapper_kwargs
            )
        else:
            self.proposal_base_strategy_model = self.base_strategy_model
        
        
        #合理行动采样器初始化
        #产生的数据结构：

        # 行动生成器，每个国家最多考虑 8 个合理行动
        # 与蓝图模型绑定，用于生成初始行动集
        # 配置文件中的内容：
        # plausible_orders_cfg {
        #     n_plausible_orders: 8
        #     max_actions_units_ratio: 3
        #     req_size: 2100
        # }
        self.order_sampler = PlausibleOrderSampler(
            cfg.plausible_orders_cfg,
            base_strategy_model=self.proposal_base_strategy_model,
            parlai_model_cfg=cfg.parlai_model_orders,
        )
        self.order_aug_cfg = cfg.order_aug

        #无
        if cfg.rescoring_blueprint_model_path:
            assert self.parlai_model_orders is None
            assert (
                cfg.order_aug.do is None
            ), "Cannot use DO with rescoring_blueprint_model_path. Use multibp rescoring instead"
            self.rescoring_blueprint_model = BaseStrategyModelWrapper(
                cfg.rescoring_blueprint_model_path, **base_strategy_model_wrapper_kwargs
            )
        else:
            self.rescoring_blueprint_model = None

        self.exploited_agent = None
        self.exploited_agent_power = None
        self.exploited_agent_num_samples = 1
        if cfg.exploited_searchbot_cfg is not None and cfg.exploited_agent_power:
            # When exploiting an agent, we run a one-sided regret minimization with full knowledge of the fixed exploited
            # policy. So we need to make sure we have their actual policy.
            exploited_searchbot_cfg = cfg.exploited_searchbot_cfg.to_editable()
            logging.info(
                f"Replacing exploited agent device {exploited_searchbot_cfg.device} -> {cfg.device}"
            )
            exploited_searchbot_cfg.device = cfg.device
            exploited_searchbot_cfg = exploited_searchbot_cfg.to_frozen()
            assert not exploited_searchbot_cfg.use_final_iter
            assert cfg.exploited_agent_power in POWERS, cfg.exploited_agent_power
            self.exploited_agent = SearchBotAgent(exploited_searchbot_cfg)
            self.exploited_agent_power = cfg.exploited_agent_power
            self.exploited_agent_num_samples = cfg.exploited_agent_num_samples
            logging.info(
                f"Exploited agent: {self.exploited_agent_power} {exploited_searchbot_cfg}"
            )

        self.log_intermediate_iterations = cfg.log_intermediate_iterations
        self.log_bilateral_values = cfg.log_bilateral_values

        self.do_incremental_search = cfg.do_incremental_search
        logging.info(f"Initialized SearchBot Agent: {self.__dict__}")

    def initialize_state(self, power: Power) -> AgentState:
        return SearchBotAgentState(power)

    def get_exploited_agent_power(self) -> Optional[Power]:
        return self.exploited_agent_power

    def override_has_press(self, has_press: bool):
        self.has_press = has_press
        self.base_strategy_model_rollouts.override_has_press(has_press)

    # Overrides BaseAgent
    def can_share_strategy(self) -> bool:
        # # 只有在策略不依赖对话、且 qre 没有 per-power 特别参数时才允许“共享策略”
        # It's only safe to share strategy if the strategy is not conditional on dialogue.
        # If qre uses different params per power, then its not symmetric or safe
        search_can_share_strategy = (self.qre is None) or (
            self.qre is not None
            and self.qre.agent_qre_lambda is None
            and self.qre.agent_qre_entropy_factor is None
        )
        return self.parlai_model_orders is None and search_can_share_strategy

    # Overrides BaseAgent
    def get_orders(self, game: Game, power: Power, state: AgentState) -> Action:
        assert isinstance(state, SearchBotAgentState)
        # 尝试获取缓存结果（SearchBotAgentState获取）
        cfr_result = self.try_get_cached_search_result(game, state)
        # 如果有新消息 → 当前局面的信息集变了，不能直接用旧的搜索结果；
        # 如果没有新消息 → 搜索结果在信息上仍然合理，可以重复使用。

        if not cfr_result:
            # 生成蓝图策略
            bp_policy = self.maybe_get_incremental_bp(game, agent_power=power, agent_state=state)

            if self.use_br_correlated_search(game.phase, "final_order"):
                # 完全没用！
                # print("in use_br_correlated_search")
                # input()

                cfr_result = self.run_best_response_against_correlated_bilateral_search(
                    game,
                    bp_policy=bp_policy,
                    agent_power=power,
                    early_exit_for_power=power,
                    agent_state=state,
                )
            else:
                # 运行CFR搜索
                cfr_result = self.run_search(
                    game,
                    bp_policy=bp_policy,
                    agent_power=power,
                    early_exit_for_power=power,
                    agent_state=state,
                )
        state.update(game, power, cfr_result, None)
        # 从CFR结果中采样行动
        return cfr_result.sample_action(power)

    # Overrides BaseAgent
    # 没用
    def get_orders_many_powers(self, game: Game, powers: List[Power],) -> JointAction:
        assert (
            self.message_handler is None
        ), "This searchbot agent appears to be a full-press agent. Do not use get_orders_many_powers in full-press since not all agents see the same info."
        assert not self.use_final_iter, "unsafe: use_final_iter"
        cfr_result = self.run_search(game, agent_state=None,)
        return {power: cfr_result.sample_action(power) for power in powers}

    # Overrides BaseSearchAgent
    def get_plausible_orders_policy(
        self,
        game: Game,
        *,
        agent_power: Optional[Power] = None,
        agent_state: Optional[AgentState],
        player_rating: Optional[PlayerRating] = None,
        allow_augment: bool = True,
    ) -> PowerPolicies:
        """Compute blueprint policy for all agents.

        If exploiting an agent, the blueprint will be the agent's computed average policy.
        If allow_augment is false, will not attempt to apply augmentations like double oracle.
        计算所有智能体的蓝图策略。

        如果利用某个智能体，则蓝图将是该智能体计算得到的平均策略。

        如果 allow_augment 为 false，则不会尝试应用诸如双重预言机之类的增强策略。
        """

        # Determine the set of plausible actions to consider for each power确定每个国家可考虑采取的一系列合理行动。
        # self.order_sampler = PlausibleOrderSampler order_sampler是一个用蓝图策略初始化好的每个阵营的动作概率
        policy = self.order_sampler.sample_orders(
            game, agent_power=agent_power, speaking_power=agent_power, player_rating=player_rating
        )
        print("policy", policy)
        # 这里输出的是各个国家，所有单位，前八个概率的可选动作
        # 'AUSTRIA': {('A VIE - GAL', 'F TRI - ALB', 'A BUD - SER'): 0.6830951448000878, ('A VIE - TRI', 'F TRI - ALB', 'A BUD - SER'): 0.10591494933371186, ('A VIE - GAL', 'F TRI - VEN', 'A BUD - SER'): 0.08524365723758108, ('A VIE - GAL', 'F TRI S A VEN', 'A BUD - SER'): 0.039619842664801584, ('A VIE - GAL', 'F TRI H', 'A BUD - SER'): 0.03310225152952094, ('A VIE - BUD', 'F TRI - ALB', 'A BUD - SER'): 0.025802689762493107, ('A VIE - TRI', 'F TRI - ALB', 'A BUD - GAL'): 0.014610897224794223, ('A VIE - TYR', 'F TRI - ALB', 'A BUD - SER'): 0.012610567447009415}, 
        # input()
        # print("policy", policy)

        if self.rescoring_blueprint_model is not None:
            #无
            #如配置 再用另一套 BP 模型重打分。
            policy = self.order_sampler.rescore_actions_base_strategy_model(
                game,
                has_press=self.has_press,
                agent_power=agent_power,
                input_policy=policy,
                model=self.rescoring_blueprint_model.model,
            )
            self.order_sampler.log_orders(game, policy, label="AFTER BP-rescoring")

        # If we are exploiting an agent, compute their policy and replace the blueprint
        # with their known average policy.# 如果我们正在利用？？某个代理（把他当敌人吗），计算他们的策略，并将蓝图替换为他们已知的平均策略。（什么是已知平均策略）
        # print(self.exploited_agent_power)
        # input()
        # 无
        if self.exploited_agent_power is not None:
            exploited_policy = collections.defaultdict(float)
            per_sample_weight = 1.0 / self.exploited_agent_num_samples
            for i in range(self.exploited_agent_num_samples):
                # ahh this is terrible!
                assert self.exploited_agent is not None
                assert not self.exploited_agent.use_final_iter
                sample_policy = self.exploited_agent.run_search(
                    game, agent_power=self.exploited_agent_power, agent_state=None
                ).avg_policies[self.exploited_agent_power]
                for action in sample_policy:
                    exploited_policy[action] += per_sample_weight * sample_policy[action]
            # Convert defaultdict -> ordinary dict
            exploited_policy = dict(exploited_policy)
            # Replace the blueprint for the power being exploited
            policy[self.exploited_agent_power] = exploited_policy

        # Inference time double oracle or other augmentation.
        # print("allow_augment", allow_augment)
        # input()
        # 有 但扩展的啥？？
        # 这是“在推理期做动作集增强”，把原来 bp_policy 的候选集再扩一圈，以免漏掉潜在好动作。
        # print("policy_before_augment", policy)
        if allow_augment:
            with temp_redefine(self.base_strategy_model_rollouts, "max_rollout_length", 0):
                with temp_redefine(self, "cache_rollout_results", True):
                    original_policy, policy = (
                        policy,
                        augment_plausible_orders(
                            game,
                            policy,
                            self,
                            self.order_aug_cfg,
                            agent_power=agent_power,
                            limits=self.order_sampler.get_plausible_order_limits(game),
                        ),
                    )

            for power in sorted(policy):
                new_actions = set(policy[power]).difference(original_policy[power])
                for i, action in enumerate(sorted(new_actions)):
                    logging.info(
                        "Order augmentation. New order for %s (%d/%d): %s",
                        power,
                        i + 1,
                        len(new_actions),
                        action,
                    )
        # print("policy_after_augment", policy)
        # input()
        return policy
    # 实际没用到
    def use_br_correlated_search(self, phase: str, mode: str):
        assert mode in ["final_order", "pseudo_order"], mode
        if self.br_corr_bilateral_search_cfg is None:
            return False
        if "MOVEMENT" not in phase:
            return False
        if mode == "final_order" and self.br_corr_bilateral_search_cfg.enable_for_final_order:
            return True
        if mode == "pseudo_order" and self.br_corr_bilateral_search_cfg.enable_for_pseudo_order:
            return True
        return False

    def run_search(
        self,
        game: Game,
        *,
        bp_policy: Optional[PowerPolicies] = None,
        early_exit_for_power: Optional[Power] = None,
        timings: Optional[TimingCtx] = None,
        extra_plausible_orders: Optional[PlausibleOrders] = None,
        agent_power: Optional[Power] = None,
        agent_state: Optional[AgentState],
    ) -> CFRResult:
        """Computes an equilibrium policy for all powers.计算所有势力的均衡策略。

        Arguments:
            - game: Game object encoding current game state.编码当前游戏状态的游戏对象。
            - bp_policy: If set, overrides the plausible order set and blueprint policy for initialization.如果设置，则覆盖初始化时的合理顺序集和蓝图策略。
                         Values should be probabilities, but can be set to -1 to simply specify plausible orders;值应为概率，但可以设置为 -1 以仅指定合理顺序；
                         in that case, this function will raise an error if any feature uses the BP distribution (e.g. bp_iters > 0)在这种情况下，如果任何功能使用蓝图分布（例如 bp_iters > 0），则此函数将引发错误。
            - early_exit_for_power: If set, then if this power has <= 1 plausible order, will exit early without computing a full equilibrium.如果设置，则如果此势力的合理顺序小于等于 1，则会提前退出，而不计算完整的均衡。
            - timings: A TimingCtx object to measure timings用于测量时间的 TimingCtx 对象。
            - extra_plausible_orders: Extra plausible orders to add to the base_strategy_model-computed set.要添加到 base_strategy_model 计算集的额外合理顺序。
            - agent_power: Optionally, specify which agent is computing the equilibrium.可选，指定哪个代理正在计算均衡。
                           Used by parlai plausible order generation, as well as advanced features like bilateral strategy.用于 parlai 合理顺序生成以及高级策略。双边战略等特征。

        Returns:
            - CFRResult object:
                - avg_policies: {pwr: avg_policy} for each power每个大国的平均政策（pwr: avg_policy）
                - final_policies: {pwr: avg_policy} for each power
                - cfr_data: detailed internal information from the CFR procedure
        """

        # 设置 deadline（max_seconds），如果超时会提前结束搜索；
        if timings is None:
            timings = TimingCtx()
        timings.start("one-time")

        deadline: Optional[float] = (
            time.monotonic() + self.max_seconds if self.max_seconds > 0 else None
        )

        # If there are no locations to order, bail
        # 如果这个国家没有可下命令的地方，就退出
        if early_exit_for_power and len(game.get_orderable_locations()[early_exit_for_power]) == 0:
            if agent_power is not None:
                assert early_exit_for_power == agent_power
            return _early_quit_cfr_result(early_exit_for_power)

        logging.info(f"BEGINNING CFR run_search, agent_power={agent_power}")

        #如果配置允许缓存 rollout 结果，就建一个 cache 对象；后面每次 do_rollouts_maybe_cached 都会往里面查/存，避免重复算。
        maybe_rollout_results_cache = (
            self.base_strategy_model_rollouts.build_cache() if self.cache_rollout_results else None
        )

        if bp_policy is None:
            # 载入蓝图策略 如果没传 bp_policy，就调 get_plausible_orders_policy 生成一份；
            bp_policy = self.get_plausible_orders_policy(
                game,
                agent_power=agent_power,
                agent_state=agent_state,
                player_rating=self.player_rating if self.set_player_rating else None,
            )
        # extra_plausible_orders 额外的很合理命令顺序， 目前看来这个算法里没有（我们以后可以加那些主观上会用到的策略 比如三十六计之类的）
        # print("extra_plausible_orders", extra_plausible_orders)
        # input()
        if extra_plausible_orders:
            for p, orders in extra_plausible_orders.items():
                for order in orders:
                    bp_policy[p].setdefault(order, 0.0)
                logging.info(f"Adding extra plausible orders {p}: {orders}")

        # CFRData 会根据 bp_policy 初始化：
        # self.power_plausible_orders（每个国家的动作列表）；
        # 内部的 CFRStats（C++对象，存累积遗憾、累积策略等）。
        cfr_data = CFRData(
            bp_policy,
            use_optimistic_cfr=self.use_optimistic_cfr,
            qre=self.qre,
            agent_power=agent_power,
        )

        # If there a single plausible action, no need to search.
        # trivial 早退：只有一个 plausible action
        if (
            early_exit_for_power
            and len(cfr_data.power_plausible_orders[early_exit_for_power]) == 1
        ):

            [the_action] = cfr_data.power_plausible_orders[early_exit_for_power]
            # print("847early_exit_for_power", early_exit_for_power)
            # 后期有用，初始化最开始可能没用
            return _early_quit_cfr_result(early_exit_for_power, action=the_action)

        # run rollouts or get from cache
        if self.cache_rollout_results and self.precompute_cache:
            # 没用...
            # print("cache_rollout_results and precompute_cache")
            # input()
            num_active_powers = sum(
                len(actions) > 1 for actions in cfr_data.power_plausible_orders.values()
            )
            if num_active_powers > 2:
                logging.warning(
                    "Disabling precomputation of the CFR cache as have %d > 2 active powers",
                    num_active_powers,
                )
            else:
                joint_orders = sample_all_joint_orders(cfr_data.power_plausible_orders)
                self.base_strategy_model_rollouts.do_rollouts_maybe_cached(
                    game,
                    agent_power=agent_power,
                    set_orders_dicts=joint_orders,
                    cache=maybe_rollout_results_cache,
                    timings=timings,
                )
        # print("NOOOO  cache_rollout_results and precompute_cache")
        # input()

        if agent_power is not None:
            bilateral_stats = BilateralStats(game, agent_power, cfr_data.power_plausible_orders)
        else:
            bilateral_stats = None

        logging.info("Starting CFR iters...")
        last_search_iter = False
        for cfr_iter in range(self.n_rollouts):
             # 0. 超时 & 迭代日志控制
            if last_search_iter:
                logging.info(f"Early exit from CFR after {cfr_iter} iterations by timeout")
                break
            elif deadline is not None and time.monotonic() >= deadline:
                last_search_iter = True
            timings.start("start")
            # do verbose logging on 2^x iters
            verbose_log_iter = self.is_verbose_log_iter(cfr_iter) or last_search_iter

            timings.start("query_policy")
            # get policy probs for all powers

            # 1. 决定这一轮用什么策略：蓝图 / 当前 CFR 策略
            power_action_ps = self.get_cur_iter_strategies(cfr_data, cfr_iter)

            timings.start("apply_orders")
            # sample policy for all powers
            # 2. 按策略为每个势力 sample 一个 joint action
            _, power_sampled_orders = sample_orders_from_policy(
                cfr_data.power_plausible_orders, power_action_ps
            )
            if bilateral_stats is not None:
                bilateral_stats.accum_bilateral_probs(power_sampled_orders, weight=cfr_iter)##起到一个记录作用似乎
            set_orders_dicts = make_set_orders_dicts(
                cfr_data.power_plausible_orders, power_sampled_orders
            )# 这是 rollouts 的输入
            # power_plausible_orders：每个势力的动作列表
            # power_sampled_orders：从当前策略采样到的基准 joint action（动作索引）。

            timings.stop()

            #对所有 joint orders 做 rollout 评估
            all_rollout_results = self.base_strategy_model_rollouts.do_rollouts_maybe_cached(
                game,
                agent_power=agent_power,
                set_orders_dicts=set_orders_dicts,
                cache=maybe_rollout_results_cache,
                timings=timings,
                player_rating=self.player_rating,
            )
            timings.start("cfr")

            for pwr, actions in cfr_data.power_plausible_orders.items():
                # pop this power's results
                results, all_rollout_results = (
                    all_rollout_results[: len(actions)],
                    all_rollout_results[len(actions) :],
                )
                if bilateral_stats is not None:
                    bilateral_stats.accum_bilateral_values(pwr, cfr_iter, results)
                # logging.info(f"Results {pwr} = {results}")
                # calculate regrets
                action_utilities: List[float] = [r[1][pwr] for r in results]
                state_utility: float = np.dot(power_action_ps[pwr], action_utilities)  # type: ignore

                # log some action values
                if verbose_log_iter:
                    self.log_cfr_iter_state(
                        game=game,
                        pwr=pwr,
                        actions=actions,
                        cfr_data=cfr_data,
                        cfr_iter=cfr_iter,
                        state_utility=state_utility,
                        action_utilities=action_utilities,
                        power_sampled_orders=power_sampled_orders,
                    )

                # update cfr data structures
                cfr_data.update(
                    pwr=pwr,
                    actions=actions,
                    state_utility=state_utility,
                    action_utilities=action_utilities,
                    which_strategy_to_accumulate=CFRStats.ACCUMULATE_PREV_ITER,
                    cfr_iter=cfr_iter,
                )

            if self.enable_compute_nash_conv and verbose_log_iter:
                ## 无...
                # print("964 compute_nash_conv")
                # input()
                logging.info(f"Computing nash conv for iter {cfr_iter}")
                self.compute_nash_conv(
                    cfr_data,
                    f"cfr iter {cfr_iter}",
                    game,
                    cfr_data.avg_strategy,
                    maybe_rollout_results_cache,
                    agent_power=agent_power,
                )

            if maybe_rollout_results_cache is not None and verbose_log_iter:
                logging.info(f"{maybe_rollout_results_cache}")

        timings.start("to_dict")

        # return prob. distributions for each power
        # 循环结束后的收尾：生成 avg_policies / final_policies
        avg_ret, final_ret = {}, {}
        power_is_loser = self.get_power_loser_dict(cfr_data, self.n_rollouts)
        for p in POWERS:
            if power_is_loser[p] or p == self.exploited_agent_power:
                avg_ret[p] = final_ret[p] = cfr_data.bp_policy(p)
            else:
                avg_ret[p] = cfr_data.avg_policy(p)
                final_ret[p] = cfr_data.cur_iter_policy(p)

        if agent_power is not None:
            logging.info(f"Final avg strategy: {avg_ret[agent_power]}")

        logging.info(
            "Raw Values: %s",
            {
                p: f"{x:.3f}"
                for p, x in zip(
                    POWERS,
                    self.base_strategy_model.get_values(
                        game, has_press=self.has_press, agent_power=agent_power
                    ),
                )
            },
        )
        logging.info("CFR Values: %s", {p: f"{cfr_data.avg_utility(p):.3f}" for p in POWERS})

        timings.stop()

        if bilateral_stats is not None and self.log_bilateral_values:
            bilateral_stats.log(cfr_data, min_order_prob=self.bilateral_cfg.min_order_prob)

        timings.pprint(logging.getLogger("timings").info)

        return CFRResult(
            bp_policies=bp_policy,
            avg_policies=avg_ret,
            final_policies=final_ret,
            cfr_data=cfr_data,
            use_final_iter=self.use_final_iter,
            bilateral_stats=bilateral_stats,
        )

    def run_bilateral_search_with_conditional_evs(
        self, game: Game, *args, **kwargs,
    ):
        raise NotImplementedError

    def run_best_response_against_correlated_bilateral_search(
        self, game: Game, *args, **kwargs,
    ):
        raise NotImplementedError





    # 也许no-press可以不用！！
    def maybe_get_incremental_bp(
        self,
        game: Game,
        agent_power: Power,
        agent_state: SearchBotAgentState,
        extra_plausible_orders: Optional[PlausibleOrders] = None,
        parlai_req_size: int = 10,
        policy_top_n: int = -1,
    ) -> Optional[PowerPolicies]:

      # maybe_get_incremental_bp 用来在“已经有上一轮搜索结果”的情况下，复用上一次的蓝图策略，只对“被新消息影响的阵营”或者“新加的候选 order”做增量更新，从而避免每次都从零跑一遍 plausible orders+蓝图策略。

       # 也就是：如果可以增量更新，就返回一份新的蓝图 bp_policy；如果不合适增量，就返回 None，让后面直接用 get_plausible_orders_policy 从头算。

        if extra_plausible_orders is None:
            extra_plausible_orders = {}
        # 如果没有上一轮的 search 结果，或者配置里关了 do_incremental_search，
        #  直接返回 None，表示“别增量了，从头算”。

        last_search_result = agent_state.get_last_search_result(game)
        if last_search_result is None or not self.do_incremental_search:
            return None

        # If this is a "rollout" phase we can't guarantee that the game state is the same as 用 last_dialogue_phase + current_short_phase 来区分：“当前是否处在跟上次搜索同一个对话阶段的真实局面”，还是已经进入了某种 rollout / 模拟状态（状态可能和上次搜索时不同）
        # last time. We could try to be careful but lets just bail.
        # 如果对话 phase 不一样了，说明局面可能已经变了（例如时间推进、rollouts 回放过很多东西）
        # → 不敢保证上次的蓝图仍然合理
        # → 增量有风险，直接返回 None，让系统重新生成蓝图。
        last_dialogue_phase = game.get_metadata("last_dialogue_phase")
        if last_dialogue_phase and last_dialogue_phase != game.current_short_phase:
            return None

        recent_messages = agent_state.get_new_messages(game)  #把“有新消息的势力”挑出来

        powers_to_update = set([m["sender"] for m in recent_messages]) | set(
            [m["recipient"] for m in recent_messages]
        )
        powers_to_update &= set(POWERS)  # don't allow ALL
        last_bp = last_search_result.get_bp_policy() #last_bp：蓝图策略（PlausibleOrderSampler 算出来的 BP 分布）
        last_agent_policy = last_search_result.get_agent_policy() #last_agent_policy：CFR 后这个 agent 使用的策略（最重要）
        last_pop_policy = last_search_result.get_population_policy() # last_pop_policy：CFR 后“策略族”的 population policy（对别的 agent 视角）

        # only take the top N actions by probability in the search population
        # 4. “截断”一下：只保留 top-N 动作（可选）
        if policy_top_n > 0:
            last_bp_thinned = {
                pwr: {
                    a: last_bp[pwr][a]
                    for a in (
                        list(last_bp[pwr])[:policy_top_n]
                        + list(last_agent_policy[pwr])[:policy_top_n]
                        + list(last_pop_policy[pwr])[:policy_top_n]
                    )
                }
                for pwr in POWERS
            }
        else:
            last_bp_thinned = copy.deepcopy(last_bp)

        # in theory, incremental updates could keep increasing the #plausible
        # if policy_top_n is not set. We can't cut it off after incremental_update
        # because we don't want to remove the extra plausible orders.
        # # So lets just cut it off here.
        # 即使 policy_top_n 没设，多轮增量更新 + 对话加入新动作 会导致动作集合不断膨胀；但又不能在 incremental_update_policy 后随便砍掉动作（因为可能刚刚加了一些 extra orders）；所以选择在这里用 cutoff_policy 按 limits 做一次统一截断。
        limits = self.order_sampler.get_plausible_order_limits(game)
        last_bp_thinned = cutoff_policy(last_bp_thinned, limits)

        for pwr, actions in extra_plausible_orders.items():
             # 某些情况下，调用方（上层）想强行把一些动作“塞入 plausible 集合”
            for a in actions:
                if a in last_bp[pwr] and not recent_messages:
                    # can use the cached BP prob
                    last_bp_thinned[pwr][a] = last_bp[pwr][a]
                else:
                    # need to recompute the BP prob
                    last_bp_thinned[pwr][a] = 0.0
                    powers_to_update.add(pwr)

        logging.info(f"Incremental update will update orders for: {powers_to_update}")

        return self.order_sampler.incremental_update_policy(
            game,
            last_bp_thinned,
            agent_power,
            powers=list(powers_to_update),
            parlai_req_size=parlai_req_size if recent_messages else 0,
        ) # 真正的“增量更新” 只更新 powers_to_update 中的国家、如果没有新消息，就不问 ParlAI

    def try_get_cached_search_result(
        self, game: Game, state: SearchBotAgentState
    ) -> Optional[SearchResult]:
        if not state.get_new_messages(game):
            return state.get_last_search_result(game)
        return None


    def log_cfr_iter_state(
        self,
        *,
        game,
        pwr,
        actions,
        cfr_data,
        cfr_iter,
        state_utility,
        action_utilities,
        power_sampled_orders,
        ptype=None,
    ):
        power_is_loser = self.get_power_loser_dict(cfr_data, cfr_iter)
        ptype_str = f":{ptype}" if ptype else ""
        logging.info(
            f"<> [ {cfr_iter+1} / {self.n_rollouts} ] {pwr}{ptype_str} {game.phase} avg_utility={cfr_data.avg_utility(pwr):.5f} cur_utility={state_utility:.5f} "
            f"is_loser= {int(power_is_loser[pwr])}"
        )
        logging.info(f">> {pwr} cur action at {cfr_iter+1}: {power_sampled_orders[pwr]}")
        logging.info(f"     {'probs':8s}  {'bp_p':8s}  {'avg_u':8s}  {'cur_u':8s}  orders")
        action_probs: List[float] = cfr_data.avg_strategy(pwr)
        bp_probs: List[float] = cfr_data.bp_strategy(pwr)
        avg_utilities: List[float] = cfr_data.avg_action_utilities(pwr)
        sorted_metrics = sorted(
            zip(actions, action_probs, bp_probs, avg_utilities, action_utilities),
            key=lambda ac: -ac[1],
        )
        for orders, p, bp_p, avg_u, cur_u in sorted_metrics:
            logging.info(f"|>  {p:8.5f}  {bp_p:8.5f}  {avg_u:8.5f}  {cur_u:8.5f}  {orders}")


    def get_power_loser_dict(self, cfr_data, cfr_iter) -> Dict[Power, bool]:
        "Determine which powers are 'losers' and should therefore play BP"
        if cfr_iter >= self.loser_bp_iter and self.loser_bp_value > 0:
            return {
                pwr: all(u < self.loser_bp_value for u in cfr_data.avg_action_utilities(pwr))
                for pwr in POWERS
            }
        else:
            return {pwr: False for pwr in POWERS}

    def is_verbose_log_iter(self, cfr_iter) -> bool:
        "Return true if we should do verbose logging on this search iteration."
        return (
            (
                self.log_intermediate_iterations
                and cfr_iter & (cfr_iter + 1) == 0
                and cfr_iter > self.n_rollouts / 8
            )
            or cfr_iter == self.n_rollouts - 1
            or (self.log_intermediate_iterations and (cfr_iter + 1) == self.bp_iters)
        )

    def get_cur_iter_strategies(
        self, cfr_data: CFRData, cfr_iter: int
    ) -> Dict[Power, List[float]]:
        "Get the current strategy for each power; either CFR"
        power_is_loser = self.get_power_loser_dict(cfr_data, cfr_iter)
        return {
            pwr: (
                cfr_data.bp_strategy(pwr)
                if (
                    cfr_iter < self.bp_iters
                    or np.random.rand() < self.bp_prob  # type:ignore
                    or power_is_loser[pwr]
                    or pwr == self.exploited_agent_power
                )
                else cfr_data.cur_iter_strategy(pwr)
            )
            for pwr in cfr_data.power_plausible_orders
        }



def augment_plausible_orders(
    game: Game,
    power_plausible_orders: PowerPolicies,
    agent: SearchBotAgent,
    cfg: agents_cfgs.SearchBotAgent.PlausibleOrderAugmentation,
    agent_power: Optional[Power] = None,
    *,
    limits: List[int],
) -> PowerPolicies:
    policy_model = agent.base_strategy_model.model
    augmentation_type = cfg.which_augmentation_type
    # print("augmentation_type", augmentation_type)
    # input()
    # searchbot的初始augmentation_type是None
    # 似乎整个项目也没扩展...搜了augmentation_type 只有一个agents.proto有这个参数
    if augmentation_type is None:
        return power_plausible_orders
    if not game.current_short_phase.endswith("M"):

        return power_plausible_orders

    if augmentation_type == "do":
        aug_cfg = cfg.do
        policy, _, _ = fairdiplomacy.action_exploration.double_oracle(
            game,
            agent,
            double_oracle_cfg=aug_cfg,
            initial_plausible_orders_policy=power_plausible_orders,
            agent_power=agent_power,
        )
        return policy

    assert augmentation_type == "random"
    aug_cfg = cfg.random

    # Creating a copy.
    power_plausible_orders = dict(power_plausible_orders)

    for power in game.get_alive_powers():
        if power == agent.exploited_agent_power:
            continue
        actions = fairdiplomacy.action_generation.generate_order_by_column_from_base_strategy_model(
            policy_model, game, selected_power=power, agent_power=agent_power
        )
        logging.info(
            "Found %s actions for %s. Not in plausible: %s",
            len(actions),
            power,
            len(frozenset(actions).difference(power_plausible_orders[power])),
        )
        max_actions = limits[POWERS.index(power)]
        # Creating space for new orders.
        orig_size = len(power_plausible_orders[power])
        power_plausible_orders[power] = dict(
            collections.Counter(power_plausible_orders[power]).most_common(
                max(aug_cfg.min_actions_to_keep, max_actions - aug_cfg.max_actions_to_drop)
            )
        )
        random.shuffle(actions)
        logging.info("Addding extra plausible orders for %s", power)
        if orig_size != len(power_plausible_orders[power]):
            logging.info(
                " (deleted %d least probable actions)",
                orig_size - len(power_plausible_orders[power]),
            )
        for action in actions:
            if len(power_plausible_orders[power]) >= max_actions:
                break
            if action not in power_plausible_orders[power]:
                power_plausible_orders[power][action] = 0
                logging.info("       %s", action)

    renormalize_policy(power_plausible_orders)

    return power_plausible_orders


def sample_all_joint_orders(power_actions: Dict[Power, List[Action]]) -> List[Dict[Power, Action]]:
    power_actions = dict(power_actions)
    for pwr in list(power_actions):
        if not power_actions[pwr]:
            power_actions[pwr] = [tuple()]

    all_orders = []
    powers, action_sets = zip(*power_actions.items())
    for joint_action in itertools.product(*action_sets):
        all_orders.append(dict(zip(powers, joint_action)))
    return all_orders


def mean(L: List[float], eps=0.0):
    return sum(L) / (len(L) + eps)




if __name__ == "__main__":
    import pathlib
    import heyhi

    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

    np.random.seed(0)  # type:ignore
    torch.manual_seed(0)  # type: ignore

    game = Game()
    cfg = heyhi.load_config(
        pathlib.Path(__file__).resolve().parents[2]
        / "conf/common/agents/searchbot_03_fastbot_loser.prototxt",
        overrides=["searchbot.n_rollouts=64"],
    )
    agent = SearchBotAgent(cfg.searchbot)
    print(agent.get_orders(game, power="AUSTRIA", state=agent.initialize_state(power="AUSTRIA")))
