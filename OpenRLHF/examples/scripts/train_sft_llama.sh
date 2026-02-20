set -x
cd
source /volume/pt-train/users/wzhang/ghchen/laip/miniconda3/etc/profile.d/conda.sh

cd /volume/pt-train/users/wzhang/ghchen/laip/code/OpenRLHF
conda activate openrlhf

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=5,7

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset /volume/pt-train/users/wzhang/ghchen/laip/correct_sft_data_gpt_oss_120b.jsonl \
   --input_key prompt \
   --output_key gpt-oss-120b-response \
   --train_batch_size 64 \
   --micro_train_batch_size 4 \
   --input_template '{}' \
   --max_samples 500000 \
   --pretrain /volume/pt-train/users/wzhang/ghchen/laip/saves/sft/qwen3_4b_sft_correct_gpt_oss_120b_epoch1 \
   --save_path /volume/pt-train/users/wzhang/ghchen/laip/saves/sft/qwen3_4b_sft_correct_gpt_oss_120b_epoch2 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-5 \
   --load_checkpoint \
   --gradient_checkpointing \
   --wandb_run_name qwen3_4b_gpt_oss_120b_epoch1 \
   --use_wandb 8ab092da28de79c21207a141b3f6f683cca27932 \
   --packing_samples
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    RANDOM_PORT=$((RANDOM % 1000 + 20000))
    deepspeed --master_port $RANDOM_PORT --module $training_commands
fi