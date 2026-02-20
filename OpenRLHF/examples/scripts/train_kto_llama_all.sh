cd
source /volume/pt-train/users/wzhang/ghchen/laip/miniconda3/etc/profile.d/conda.sh
cd /volume/pt-train/users/wzhang/ghchen/laip/code/OpenRLHF
conda activate openrlhf
export WANDB_MODE=offline
set -x
read -r -d '' training_commands <<EOF
openrlhf.cli.train_kto \
   --save_path /volume/pt-train/users/wzhang/ghchen/laip/saves/Llama-3-Base-8B-SFT/Llama-3-Base-8B-SFT_kto_all \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --pretrain /volume/pt-train/users/wzhang/ghchen/laip/models/Llama-3-Base-8B-SFT \
   --bf16 \
   --max_epochs 1 \
   --max_len 2048 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --dataset /volume/pt-train/users/wzhang/ghchen/laip/datasets/all_uf \
   --input_key instruction \
   --output_key response \
   --flash_attn \
   --beta 0.1 \
   --max_samples 1024 \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
