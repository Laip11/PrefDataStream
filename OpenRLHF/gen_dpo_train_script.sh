#!/bin/bash
# =================================================================
# 这是一个批量生成 DPO 训练脚本的启动器。
#
# 它会循环遍历下面定义的所有模型和数据集组合，
# 并为每一种组合调用 create_dpo_script.py 生成一个独立的训练脚本。
#
# 使用方法:
# 1. 在下面的 MODEL_PATHS 和 DATASET_PATHS 数组中填入你的路径。
# 2. 运行脚本: bash run_batch_generator.sh
# =================================================================

set -e # 如果任何命令失败，则立即退出脚本

# --- 在这里配置你的路径和参数 ---
cd /volume/pt-train/users/wzhang/ghchen/laip/code/OpenRLHF
# 1. 定义一个或多个模型路径的数组
MODEL_PATHS=(
# /volume/pt-train/users/wzhang/ghchen/laip/models/HuggingFaceH4/mistral-7b-sft-beta
# /volume/pt-train/users/wzhang/ghchen/laip/models/Llama-3-Base-8B-SFT
# /volume/pt-train/users/wzhang/ghchen/laip/models/Qwen2.5-7B-SFT
/volume/pt-train/models/Qwen2.5-7B-Instruct
# /volume/pt-train/users/wzhang/ghchen/laip/models/LLM-Research/Meta-Llama-3-8B-Instruct

)

# 2. 定义一个或多个数据集路径的数组
DATASET_PATHS=(
# /volume/pt-train/users/wzhang/ghchen/laip/datasets/Llama-3-Base-8B-SFT_all_uf_im_flip_15_data
# /volume/pt-train/users/wzhang/ghchen/laip/datasets/mistral-7b-sft_ad_flip_80_data
# /volume/pt-train/users/wzhang/ghchen/laip/datasets/mistral-7b-sft_ad_flip_100_data
# /volume/pt-train/users/wzhang/ghchen/laip/datasets/Qwen2_5-7B-SFT_all_uf_im_flip_20_data
# /volume/pt-train/users/wzhang/ghchen/laip/datasets/all_uf
# /volume/pt-train/users/wzhang/ghchen/laip/datasets/selectivedpo_data
# /volume/pt-train/users/wzhang/ghchen/laip/datasets/all_uf_flip
/volume/pt-train/users/wzhang/ghchen/laip/data/Qwen2.5-7B-Instruct_im_data

)

# 3. 定义通用的超参数
LR=5e-7
BETA=0.01
EPOCHS=1
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=2

# --- 脚本执行区 ---

# 4. 使用嵌套循环遍历所有模型和数据集的组合
echo "开始批量生成训练脚本..."
for model_path in "${MODEL_PATHS[@]}"; do
    for dataset_path in "${DATASET_PATHS[@]}"; do
        
        echo "------------------------------------------------"
        echo "正在为以下组合生成脚本:"
        echo "  - 模型: $model_path"
        echo "  - 数据集: $dataset_path"
        echo ""

        # 调用 Python 脚本，并传入当前的 model_path 和 dataset_path
        python create_dpo_script.py \
            --model_path "$model_path" \
            --dataset_path "$dataset_path" \
            --learning_rate "$LR" \
            --beta "$BETA" \
            --max_epochs "$EPOCHS" \
            --train_batch_size "$GLOBAL_BATCH_SIZE" \
            --micro_train_batch_size "$MICRO_BATCH_SIZE" \
            --shuffle_train_dataset
    done
done

echo "------------------------------------------------"
echo "generate scripts done"
