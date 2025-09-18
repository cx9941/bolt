#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口：只读取 YAML 配置并执行网格实验
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml  # 要求：环境需安装 pyyaml

from utils import (
    run_combo,
    set_paths,
)

def main():
    ap = argparse.ArgumentParser(description="Run grid experiments (YAML-only).")
    ap.add_argument("--config", type=str, default="configs/grid.yaml", help="YAML config path")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] YAML 配置文件不存在：{cfg_path}")
        sys.exit(1)

    with cfg_path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    # --- 必要字段检查（不提供默认，全部依赖 YAML） ---
    try:
        maps = y["maps"]                     # task -> [methods]
        methods = y["methods"]               # 参与的 method 列表
        datasets = y["datasets"]             # 数据集列表
        grid = y["grid"]                     # 网格参数字典
        run = y["run"]                       # 运行参数（gpus, max_workers, dry_run, only_collect）
        paths = y["paths"]                   # 路径（results_dir, logs_dir）
    except KeyError as e:
        print(f"[ERR] YAML 缺少必要字段：{e}")
        sys.exit(1)

    knowns = grid["known_cls_ratio"]
    labeleds = grid["labeled_ratio"]
    fold_idxs = grid["fold_idxs"]
    seeds = grid["seeds"]
    cfs = grid["cluster_num_factor"]
    
    # ★ 新增：全局 epoch 控制（必填）
    num_pretrain_epochs = int(run["num_pretrain_epochs"])
    num_train_epochs = int(run["num_train_epochs"])

    gpus = run["gpus"]            # 例如：[0,1,2] 或 []
    max_workers = int(run["max_workers"])
    dry_run = bool(run.get("dry_run", False))
    only_collect = bool(run.get("only_collect", False))

    # 设置结果/日志路径（由 YAML 指定）
    set_paths(paths["results_dir"], paths["logs_dir"])

    # 过滤：只保留在 maps 出现过的方法
    method2task = {m: t for t, ms in maps.items() for m in ms}
    final_methods = [m for m in methods if m in method2task]
    if not final_methods:
        print("[ERR] methods 为空或不在 maps 中。")
        sys.exit(1)

    if not datasets:
        print("[ERR] datasets 为空。")
        sys.exit(1)

    # 组合与并行
    combos = []
    for m in final_methods:
        for d in datasets:
            for kr in knowns:
                for lr in labeleds:
                    for fi in fold_idxs:
                        for sd in seeds:
                            for cf in cfs:
                                combos.append((m, d, kr, lr, fi, sd, cf))

    print(f"[INFO] 组合数={len(combos)} | methods={final_methods} | datasets={datasets}")
    gpu_cycle = list(gpus) if gpus else [None]

    futs = []
    from cli import METHOD_REGISTRY  # 延迟导入以避免循环
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        for idx, (m, d, kr, lr, fi, sd, cf) in enumerate(combos):
            gpu_id = gpu_cycle[idx % len(gpu_cycle)] if gpu_cycle else None
            futs.append(ex.submit(
                run_combo,
                method=m, dataset=d, known=kr, labeled=lr,
                fold_idx=fi, seed=sd, c_factor=cf,
                gpu_id=gpu_id,
                # ★ 新增参数传入
                num_pretrain_epochs=num_pretrain_epochs,
                num_train_epochs=num_train_epochs,
                dry_run=dry_run, only_collect=only_collect
            ))
        for fu in as_completed(futs):
            _ = fu.result()

    from utils import SUMMARY_CSV
    print("[DONE] 汇总文件：", SUMMARY_CSV)

if __name__ == "__main__":
    main()