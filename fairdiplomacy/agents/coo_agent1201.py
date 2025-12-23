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

LOG = logging.getLogger(__name__)


class CooAgent(BQRE1PAgent):
    """
    Cicero 价值查看器 Agent
    """

    def __init__(self, cfg, **kwargs) -> None:
        # cfg 里已经包含 base_strategy_model { model_path: "...", temperature: ... }
        # 父类会用它自动加载 blueprint 模型
        super().__init__(cfg, **kwargs)
        LOG.info("[CooAgent] 初始化完成，已加载 Cicero 价值模型和蓝图策略")

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
    cfg_path: str = "conf/common/agents/cicero_noParl.prototxt",
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
    if hasattr(full_cfg, "agent") and hasattr(full_cfg.agent, "bqre1p"):
        agent_cfg = full_cfg.agent.bqre1p
        LOG.info("[load_cicero] 使用 full_cfg.agent.bqre1p")
    elif hasattr(full_cfg, "bqre1p"):
        agent_cfg = full_cfg.bqre1p
        LOG.info("[load_cicero] 使用 full_cfg.bqre1p")
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
if __name__ == "__main__":
    """独立运行示例"""
    import argparse

    # logging.basicConfig(level=logging.INFO)
    # ##
    logging.basicConfig(level=logging.WARNING)


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
        default="conf/common/agents/cicero_noParl.prototxt",
        help="Cicero配置文件路径",
    )
    parser.add_argument(
        "--power",
        type=str,
        default="FRANCE",
        help="势力名称",
    )
    args = parser.parse_args()

    # 1) 用统一入口加载 Cicero + blueprint
    print(f"\n[1/2] 加载 Cicero Agent: {args.cfg}")
    try:
        agent = load_cicero(
            cfg_path=args.cfg,
            skip_cache=False,
        )
        print("  ✓ Agent 初始化成功")
    except Exception as e:
        print(f"  ✗ Agent 初始化失败: {e}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

    # 2) 起一盘游戏，查看指定势力在当前局面下的动作价值
    print(f"\n[2/2] 评估动作价值...")
    game = pydipcc.Game()

    try:
        action_values = agent.evaluate_action_values(
            game,
            agent_power=args.power,
        )
        print(f"\n✓ 完成！共找到 {len(action_values)} 个动作\n")
    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
