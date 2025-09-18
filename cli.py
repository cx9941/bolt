# -*- coding: utf-8 -*-
"""
按模型定制的 CLI 生成器与注册表
"""
from __future__ import annotations
import os
import sys
from typing import Any, Dict, List, Callable

CliBuilder = Callable[[Dict[str,Any], int], List[str]]  # args_json, stage_idx -> CLI list

def _common_env(args_json:Dict[str,Any]) -> List[str]:
    return [
        "--config", str(args_json["config"]),
        "--dataset", args_json["dataset"],
        "--known_cls_ratio", str(args_json["known_cls_ratio"]),
        "--seed", str(args_json["seed"]),
    ]

def _epoch_flags(args_json:Dict[str,Any], is_pretrain: bool) -> List[str]:
    """根据阶段返回统一的 epoch 参数"""
    return ["--num_pretrain_epochs", str(args_json["num_pretrain_epochs"]), "--num_train_epochs", str(args_json["num_train_epochs"])]
    if is_pretrain:
        return ["--num_pretrain_epochs", str(args_json["num_pretrain_epochs"])]
    else:
        return ["--num_train_epochs", str(args_json["num_train_epochs"])]
    

def cli_tan(args_json:Dict[str,Any], stage:int) -> List[str]:
    pre = f'./outputs/gcd/tan/premodel_{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_fold{args_json["fold_idx"]}_seed{args_json["seed"]}'
    return [
        sys.executable, "code/gcd/baselines/TAN/run.py",
        *_common_env(args_json),
        "--gpu_id", str(args_json["gpu_id"]),
        "--pretrain_dir", pre,
        "--pretrain",
        *_epoch_flags(args_json, is_pretrain=True),    # ★ 新增
        "--save_model",
        "--freeze_bert_parameters",
    ]
def cli_loop(args_json:Dict[str,Any], stage:int) -> List[str]:
    pre = f'outputs/gcd/loop/premodel_{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_fold{args_json["fold_idx"]}_seed{args_json["seed"]}'
    save = f'outputs/gcd/loop/model_{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_fold{args_json["fold_idx"]}_seed{args_json["seed"]}'
    return [
        sys.executable, "code/gcd/baselines/LOOP/run.py",
        *_common_env(args_json),
        "--labeled_ratio", str(args_json["labeled_ratio"]),
        "--fold_idx", str(args_json["fold_idx"]),
        "--gpu_id", str(args_json["gpu_id"]),
        "--pretrain_dir", pre,
        "--save_model_path", save,
        *_epoch_flags(args_json, is_pretrain=False),   # ★ 新增
        "--save_premodel",
        "--save_model",
    ]

def cli_glean(args_json:Dict[str,Any], stage:int) -> List[str]:
    pre = f'./outputs/gcd/glean/premodel_{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_fold{args_json["fold_idx"]}_seed{args_json["seed"]}'
    save = f'./outputs/gcd/glean/model_{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_fold{args_json["fold_idx"]}_seed{args_json["seed"]}'
    cli = [
        sys.executable, "code/gcd/baselines/Glean/run.py",
        *_common_env(args_json),
        "--labeled_ratio", str(args_json["labeled_ratio"]),
        "--fold_idx", str(args_json["fold_idx"]),
        "--gpu_id", str(args_json["gpu_id"]),
        *_epoch_flags(args_json, is_pretrain=False),   # ★ 新增
        "--save_premodel",
        "--save_model",
        "--feedback_cache",
        "--flag_demo",
        "--flag_filtering",
        "--flag_demo_c",
        "--flag_filtering_c",
        "--pretrain_dir", pre,
        "--save_model_path", save,
    ]
    # 透传 OpenAI key（不拼进命令，脚本内通过环境读取）
    if "OPENAI_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    return cli

def cli_geoid(args_json:Dict[str,Any], stage:int) -> List[str]:
    return [
        sys.executable, "code/gcd/baselines/GeoID/run.py",
        *_common_env(args_json),
        "--labeled_ratio", str(args_json["labeled_ratio"]),
        "--fold_idx", str(args_json["fold_idx"]),
        "--seed", str(args_json["seed"]),
        *_epoch_flags(args_json, is_pretrain=False),   # ★ 新增
        "--report_pretrain",
    ]

def cli_dpn(args_json:Dict[str,Any], stage:int) -> List[str]:
    return [
        sys.executable, "code/gcd/baselines/DPN/run.py",
        *_common_env(args_json),
        "--labeled_ratio", str(args_json["labeled_ratio"]),
        "--fold_idx", str(args_json["fold_idx"]),
        "--gpu_id", str(args_json["gpu_id"]),
        "--freeze_bert_parameters",
        "--save_model",
        "--pretrain",
    ]

def cli_deepaligned(args_json:Dict[str,Any], stage:int) -> List[str]:
    return [
        sys.executable, "code/gcd/baselines/DeepAligned-Clustering/run.py",
        *_common_env(args_json),
        "--labeled_ratio", str(args_json["labeled_ratio"]),
        "--fold_idx", str(args_json["fold_idx"]),
        "--gpu_id", str(args_json["gpu_id"]),
        "--freeze_bert_parameters",
        *_epoch_flags(args_json, is_pretrain=True),    # ★ 新增
        "--save_model",
        "--pretrain",
    ]

def cli_alup(args_json:Dict[str,Any], stage:int) -> List[str]:
    base = f'{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_fold{args_json["fold_idx"]}_{args_json["seed"]}'
    pre_sub = f"pretrain/{base}"
    fin_sub = f"finetune/{base}"
    if stage == 1:
        return [
            sys.executable, "code/gcd/baselines/ALUP/run.py",
            *_common_env(args_json),
            "--labeled_ratio", str(args_json["labeled_ratio"]),
            "--fold_idx", str(args_json["fold_idx"]),
            "--gpu_id", str(args_json["gpu_id"]),
            "--do_pretrain_and_contrastive",
            *_epoch_flags(args_json, is_pretrain=True),    # ★ 新增
            "--output_subdir", pre_sub,
        ]
    else:
        return [
            sys.executable, "code/gcd/baselines/ALUP/run.py",
            *_common_env(args_json),
            "--labeled_ratio", str(args_json["labeled_ratio"]),
            "--fold_idx", str(args_json["fold_idx"]),
            "--gpu_id", str(args_json["gpu_id"]),
            "--do_al_finetune",
            *_epoch_flags(args_json, is_pretrain=False),   # ★ 新增
            "--pretrained_stage1_subdir", pre_sub,
            "--output_subdir", fin_sub,
            "--save_results",
        ]

def cli_sdc_pre(args_json:Dict[str,Any], stage:int) -> List[str]:
    pre = f'outputs/gcd/sdc/premodels/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_fold{args_json["fold_idx"]}_seed{args_json["seed"]}'
    return [
        sys.executable, "code/gcd/baselines/SDC/pretrain.py",
        *_common_env(args_json),
        "--labeled_ratio", str(args_json["labeled_ratio"]),
        "--fold_idx", str(args_json["fold_idx"]),
        "--gpu_id", str(args_json["gpu_id"]),
        "--pretrain_dir", pre,
        *_epoch_flags(args_json, is_pretrain=True),        # ★ 新增
        "--pretrain",
        "--save_model",
    ]

def cli_sdc_run(args_json:Dict[str,Any], stage:int) -> List[str]:
    pre = f'outputs/gcd/sdc/premodels/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_fold{args_json["fold_idx"]}_seed{args_json["seed"]}'
    train = f'outputs/gcd/sdc/models/{args_json["dataset"]}_{args_json["known_cls_ratio"]}_{args_json["labeled_ratio"]}_fold{args_json["fold_idx"]}_seed{args_json["seed"]}'
    return [
        sys.executable, "code/gcd/baselines/SDC/run.py",
        *_common_env(args_json),
        "--labeled_ratio", str(args_json["labeled_ratio"]),
        "--fold_idx", str(args_json["fold_idx"]),
        "--gpu_id", str(args_json["gpu_id"]),
        "--pretrain_dir", pre,
        "--train_dir", train,
        *_epoch_flags(args_json, is_pretrain=False),       # ★ 新增
        "--save_model",
    ]

def cli_plm_gcd(args_json:Dict[str,Any], stage:int) -> List[str]:
    return [
        sys.executable, "code/gcd/plm_gcd/run.py",
        *_common_env(args_json),
        "--labeled_ratio", str(args_json["labeled_ratio"]),
        "--fold_idx", str(args_json["fold_idx"]),
        "--gpu_id", str(args_json["gpu_id"]),
        *_epoch_flags(args_json, is_pretrain=False),       # ★ 新增
    ]

def cli_simple_openset(entry:str) -> CliBuilder:
    def _f(args_json:Dict[str,Any], stage:int) -> List[str]:
        return [
            sys.executable, entry,
            "--config", str(args_json["config"]),
            "--dataset", args_json["dataset"],
            "--known_cls_ratio", str(args_json["known_cls_ratio"]),
            "--labeled_ratio", str(args_json["labeled_ratio"]),
            "--fold_idx", str(args_json["fold_idx"]),
            "--seed", str(args_json["seed"]),
            "--gpu_id", str(args_json["gpu_id"]),
            *_epoch_flags(args_json, is_pretrain=False),    # ★ 新增
        ]
    return _f

# --- 注册表：方法 -> {task, stages:[{entry, cli_builder}], config, output_base} ---
METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "tan": {
        "task":"gcd",
        "stages":[{"entry":"code/gcd/baselines/TAN/run.py", "cli_builder": cli_tan}],
        "config":"configs/gcd/tan.yaml",
        "output_base":"./outputs/gcd/tan",
    },
    "loop": {
        "task":"gcd",
        "stages":[{"entry":"code/gcd/baselines/LOOP/run.py", "cli_builder": cli_loop}],
        "config":"configs/gcd/loop.yaml",
        "output_base":"./outputs/gcd/loop",
    },
    "glean": {
        "task":"gcd",
        "stages":[{"entry":"code/gcd/baselines/Glean/run.py", "cli_builder": cli_glean}],
        "config":"configs/gcd/glean.yaml",
        "output_base":"./outputs/gcd/glean",
    },
    "geoid": {
        "task":"gcd",
        "stages":[{"entry":"code/gcd/baselines/GeoID/run.py", "cli_builder": cli_geoid}],
        "config":"configs/gcd/geoid.yaml",
        "output_base":"./outputs/gcd/geoid",
    },
    "dpn": {
        "task":"gcd",
        "stages":[{"entry":"code/gcd/baselines/DPN/run.py", "cli_builder": cli_dpn}],
        "config":"configs/gcd/dpn.yaml",
        "output_base":"./outputs/gcd/dpn",
    },
    "deepaligned": {
        "task":"gcd",
        "stages":[{"entry":"code/gcd/baselines/DeepAligned-Clustering/run.py", "cli_builder": cli_deepaligned}],
        "config":"configs/gcd/deepaligned.yaml",
        "output_base":"./outputs/gcd/deepaligned",
    },
    "alup": {
        "task":"gcd",
        "stages":[
            {"entry":"code/gcd/baselines/ALUP/run.py", "cli_builder": cli_alup},  # stage 1
            {"entry":"code/gcd/baselines/ALUP/run.py", "cli_builder": cli_alup},  # stage 2
        ],
        "config":"configs/gcd/alup.yaml",
        "output_base":"./outputs/gcd/alup",
    },
    "sdc": {
        "task":"gcd",
        "stages":[
            {"entry":"code/gcd/baselines/SDC/pretrain.py", "cli_builder": cli_sdc_pre},
            {"entry":"code/gcd/baselines/SDC/run.py",      "cli_builder": cli_sdc_run},
        ],
        "config":"configs/gcd/sdc.yaml",
        "output_base":"./outputs/gcd/sdc",
    },
    "plm_gcd": {
        "task":"gcd",
        "stages":[{"entry":"code/gcd/plm_gcd/run.py", "cli_builder": cli_plm_gcd}],
        "config":"configs/gcd/plm_gcd.yaml",
        "output_base":"./outputs/gcd/plm_gcd",
    },
    # ----------------- openset -----------------
    "ab":      {"task":"openset","stages":[{"entry":"code/openset/baselines/AB/run.py",   "cli_builder": cli_simple_openset("code/openset/baselines/AB/run.py")}],   "config":"configs/openset/ab.yaml",     "output_base":"./outputs/openset/ab"},
    "adb":     {"task":"openset","stages":[{"entry":"code/openset/baselines/ADB/run.py",  "cli_builder": cli_simple_openset("code/openset/baselines/ADB/run.py")}],  "config":"configs/openset/adb.yaml",    "output_base":"./outputs/openset/adb"},
    "clap":    {"task":"openset","stages":[{"entry":"code/openset/baselines/CLAP/run.py", "cli_builder": cli_simple_openset("code/openset/baselines/CLAP/run.py")}], "config":"configs/openset/clap.yaml",   "output_base":"./outputs/openset/clap"},
    "deepunk": {"task":"openset","stages":[{"entry":"code/openset/baselines/DeepUNK/run.py","cli_builder": cli_simple_openset("code/openset/baselines/DeepUNK/run.py")}], "config":"configs/openset/deepunk.yaml","output_base":"./outputs/openset/deepunk"},
    "doc":     {"task":"openset","stages":[{"entry":"code/openset/baselines/DOC/run.py",  "cli_builder": cli_simple_openset("code/openset/baselines/DOC/run.py")}],  "config":"configs/openset/doc.yaml",    "output_base":"./outputs/openset/doc"},
    "dyen":    {"task":"openset","stages":[{"entry":"code/openset/baselines/Dyen/run.py", "cli_builder": cli_simple_openset("code/openset/baselines/Dyen/run.py")}], "config":"configs/openset/dyen.yaml",   "output_base":"./outputs/openset/dyen"},
    "knncon":  {"task":"openset","stages":[{"entry":"code/openset/baselines/KNNCon/run.py","cli_builder": cli_simple_openset("code/openset/baselines/KNNCon/run.py")}], "config":"configs/openset/knncon.yaml", "output_base":"./outputs/openset/knncon"},
    "plm_ood": {"task":"openset","stages":[{"entry":"code/openset/plm_ood/run.py",        "cli_builder": cli_simple_openset("code/openset/plm_ood/run.py")}],        "config":"configs/openset/plm_ood.yaml","output_base":"./outputs/openset/plm_ood"},
    "scl":     {"task":"openset","stages":[{"entry":"code/openset/baselines/SCL/run.py",  "cli_builder": cli_simple_openset("code/openset/baselines/SCL/run.py")}],  "config":"configs/openset/scl.yaml",    "output_base":"./outputs/openset/scl"},
}