# coding=utf-8
"""
coo_agent.py - 用main记载cicero，得出当下局面的可选动作及价值，已可以跑通，未验证价值是否正确
"""

from typing import Any, Dict, List, Tuple
import logging
import os

from fairdiplomacy import pydipcc
from fairdiplomacy.agents.bqre1p_agent import BQRE1PAgent

LOG = logging.getLogger(__name__)


class CooAgent(BQRE1PAgent):
    """
    Cicero价值查看器Agent
    """

    def __init__(self, cfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        LOG.info("[CooAgent] 初始化完成，已加载Cicero价值模型和蓝图策略")

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
        LOG.info(
            "[CooAgent] Phase %s, Power %s, BP策略中有 %d 个可行动作",
            current_phase,
            agent_power,
            len(our_bp_policy),
        )

        # 2. 运行Cicero搜索获取价值
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

        # 4. 打印结果
        print(f"\n{'='*70}")
        print(f"[CooAgent] Cicero动作价值分析")
        print(f"  Power: {agent_power}")
        print(f"  Phase: {current_phase}")
        print(f"{'='*70}")
        for idx, (action, value) in enumerate(action_values):
            print(f"  [{idx:2d}] 价值={value:.4f} | 动作={action}")
        print(f"{'='*70}\n")

        return action_values


if __name__ == "__main__":
    """独立运行示例"""
    import argparse
    import heyhi

    logging.basicConfig(level=logging.INFO)
    
    # 确保在项目根目录运行
    project_root = "/workspace/Diplomacy/diplomacy_cicero"
    current_dir = os.getcwd()
    
    if not current_dir.endswith("diplomacy_cicero"):
        if os.path.exists(project_root):
            os.chdir(project_root)
            print(f"[路径修复] 工作目录从 {current_dir} 切换到 {os.getcwd()}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="conf/common/agents/cicero.prototxt",
        help="Cicero配置文件路径",
    )
    parser.add_argument(
        "--power",
        type=str,
        default="FRANCE",
        help="势力名称",
    )
    args = parser.parse_args()

    # 加载配置
    print(f"\n[1/3] 加载配置: {args.cfg}")
    full_cfg = heyhi.load_config(args.cfg)
    
    # 提取bqre1p子配置
    if hasattr(full_cfg, 'agent') and hasattr(full_cfg.agent, 'bqre1p'):
        agent_cfg = full_cfg.agent.bqre1p
        print("  ✓ 成功提取 agent.bqre1p 配置")
    elif hasattr(full_cfg, 'bqre1p'):
        agent_cfg = full_cfg.bqre1p
        print("  ✓ 成功提取 bqre1p 配置")
    else:
        raise ValueError(
            f"配置文件格式错误！期望结构: agent.bqre1p 或 bqre1p\n"
            f"可用字段: {dir(full_cfg)}"
        )

    # 初始化Agent
    print(f"\n[2/3] 初始化CooAgent...")
    try:
        agent = CooAgent(
            agent_cfg,
            skip_base_strategy_model_cache=False
        )
        print("  ✓ Agent初始化成功")
    except Exception as e:
        print(f"  ✗ Agent初始化失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # 创建初始局面并评估
    print(f"\n[3/3] 评估动作价值...")
    game = pydipcc.Game()
    
    try:
        # ✅ 不再传 agent_state=None，让函数内部创建
        action_values = agent.evaluate_action_values(
            game, 
            agent_power=args.power
        )
        
        print(f"\n✓ 完成！共找到 {len(action_values)} 个动作")
    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)