# coding=utf-8
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple


# ✅ 这里假设你的 ConsistentAgent / load_cicero / get_territory_parts / POWERS
# 都在同目录的 consistent_agent.py 里（文件名你可按实际改）
# from consistent_agent_V2 import POWERS, get_territory_parts, load_consistent_agent
from consistent_agent_V3 import POWERS, get_territory_parts, load_consistent_agent
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
    parser.add_argument("--mode", type=str, default="top1", choices=["top1", "sample", "bqre"])
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
    # agent = load_cicero(args.cfg, skip_cache=False)
    agent = load_consistent_agent(args.cfg, skip_cache=False)
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
                # orders, items, used_source, dropped = agent.choose_orders(
                #     game=game,
                #     power=pwr,
                #     agent_state=states[pwr],
                #     source=args.source,
                #     top_k=args.topk,
                #     mode=args.mode,
                # )
                # tmp_orders[pwr] = orders
                # all_infos[pwr] = (items, used_source, dropped)
                info = agent.get_orders_info(
                    game=game,
                    power=pwr,
                    state=states[pwr],
                    source=args.source,
                    top_k=args.topk,
                    mode=args.mode,
                )
                orders = info["orders"]
                items = info["items"]
                used_source = info["used_source"]
                dropped = info["dropped"]
                raw_items = info.get("raw_items", []) or []
                repair_logs = info.get("repair_logs", []) or []

                tmp_orders[pwr] = orders
                all_infos[pwr] = (items, used_source, dropped, raw_items, repair_logs)

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
                items, used_source, dropped, raw_items, repair_logs = all_infos[pwr]
                f.write(f"[AGENT] power={pwr} used_source={used_source} topk={len(items)}\n")

                if repair_logs:
                    f.write(f"[REPAIR] power={pwr} n={len(repair_logs)}\n")
                    for j, ev in enumerate(repair_logs):
                        f.write(
                            f"  {ev['tag']}-{j:02d} "
                            f"dest='{ev.get('dest', '')}' "
                            f"replaced='{ev.get('replaced', '')}' "
                            f"modified_to='{ev.get('modified_to', '')}'\n"
                        )
                        f.write(f"    before={list(ev.get('before_action', []))}\n")
                        f.write(f"    after ={list(ev.get('after_action', []))}\n")

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