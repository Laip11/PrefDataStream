set -x
cd
source /volume/pt-train/users/wzhang/ghchen/laip/miniconda3/etc/profile.d/conda.sh

cd /volume/pt-train/users/wzhang/ghchen/laip/code/OpenRLHF
conda activate openrlhf
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset /volume/pt-train/users/wzhang/ghchen/laip/half_pair_data-sft-gptoss120b.jsonl \
   --input_key prompt \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --max_samples 500000 \
   --pretrain /volume/pt-train/models/Qwen3-8B \
   --save_path /volume/pt-train/users/wzhang/ghchen/laip/saves/sft/qwen3_8b_sft_half_pair_data_gpt_oss_120b \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --wandb_run_name qwen3_8b_gpt_oss_120b_epoch1 \
   --use_wandb 8ab092da28de79c21207a141b3f6f683cca27932
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    RANDOM_PORT=$((RANDOM % 1000 + 20000))
    deepspeed --master_port $RANDOM_PORT --module $training_commands
fi