#!/bin/bash

# --- 配置区 ---

# 1. 定义日志文件名，包含日期和时间，避免覆盖
LOG_FILE="run_log_$(date +'%Y-%m-%d_%H-%M-%S').txt"

# 2. 定义需要按顺序执行的脚本列表
#    请根据您的图片和需求，在这里添加所有需要运行的sh文件路径
SCRIPTS_TO_RUN=(
    "scripts/gcd/dpn.sh"
    "scripts/gcd/sdc.sh"
    "scripts/openset/ab.sh"
    "scripts/openset/adb.sh"
    "scripts/openset/clap.sh"
    "scripts/openset/doc.sh"
    "scripts/openset/dyen.sh"
    "scripts/openset/knncon.sh"
    "scripts/openset/scl.sh"
)

# --- 执行区 ---

# 初始化日志文件，并写入一个标题头
echo "======== Script Execution Log started at $(date) ========" > "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 遍历脚本列表并执行
for script in "${SCRIPTS_TO_RUN[@]}"; do
    echo "-----------------------------------------------------" | tee -a "$LOG_FILE"
    echo "INFO: Starting script: $script" | tee -a "$LOG_FILE"

    # 检查脚本文件是否存在
    if [ ! -f "$script" ]; then
        echo "ERROR: Script file not found: $script. Skipping." | tee -a "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        continue # 跳过当前循环，继续下一个
    fi

    # 记录开始时间 (使用 Unix 时间戳)
    start_time=$(date +%s)

    # 执行脚本，并将它的标准输出和错误输出都追加到日志文件中
    # 这样如果子脚本有报错，也能在日志中看到
    sh "$script" >> "$LOG_FILE" 2>&1
    
    # 获取脚本的退出码，$? 会保存上一个命令的退出码，0代表成功，非0代表失败
    exit_code=$?

    # 记录结束时间
    end_time=$(date +%s)
    
    # 计算执行耗时
    duration=$((end_time - start_time))

    # 判断是否成功并记录日志
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS: Script $script finished successfully in $duration seconds." | tee -a "$LOG_FILE"
    else
        echo "FAILURE: Script $script failed with exit code $exit_code. Skipping to the next script." | tee -a "$LOG_FILE"
    fi
    echo "" >> "$LOG_FILE"
done

echo "=====================================================" | tee -a "$LOG_FILE"
echo "All tasks completed. Log file saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "====================================================="