set -x
cd
source /volume/pt-train/users/wzhang/ghchen/laip/miniconda3/etc/profile.d/conda.sh

cd /volume/pt-train/users/wzhang/ghchen/laip/code/OpenRLHF
conda activate openrlhf

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=4,5,6,7

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 16384 \
   --dataset /volume/pt-train/users/wzhang/ghchen/laip/data/mR3_100k_data.jsonl \
   --input_key prompt \
   --output_key response \
   --train_batch_size 64 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --input_template '{}' \
   --pretrain /volume/pt-train/models/Qwen3-8B \
   --save_path /volume/pt-train/users/wzhang/ghchen/laip/saves/sft/qwen3_8b_sft_repro_100kdata \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-5 \
   --lr_warmup_ratio 0.1 \
   --load_checkpoint \
   --gradient_checkpointing \
   --wandb_run_name qwen3_8b_sft_repro_100kdata \
   --use_wandb 8ab092da28de79c21207a141b3f6f683cca27932
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi