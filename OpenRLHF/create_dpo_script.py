import argparse
import os

def create_dpo_script(
    model_path: str,
    dataset_path: str,
    sh_path: str = None,
    learning_rate: float = 5e-7,
    beta: float = 0.1,
    max_epochs: int = 1,
    train_batch_size: int = 128,
    micro_train_batch_size: int = 1,
    shuffle_train_dataset: bool = False  # 1. 新增的函数参数，默认为 False
):

    # 从输入路径中提取模型和数据集的名称，用于构造输出路径
    model_name = os.path.basename(os.path.normpath(model_path))
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # 定义检查点保存路径
    save_path = f'/volume/pt-train/users/wzhang/ghchen/laip/saves/{model_name}/{model_name}_{dataset_name}_bsz_{train_batch_size}_lr_{learning_rate}'
    os.makedirs(save_path, exist_ok=True)

    if sh_path is None:
        script_dir = '/volume/pt-train/users/wzhang/ghchen/laip/code/OpenRLHF/scripts'
        # 替换数据集名称中的斜杠，以创建有效的文件名
        dataset_filename_safe = dataset_name.replace("/", "_")
        sh_path = f'{script_dir}/train_{model_name}_{dataset_filename_safe}_bsz_{train_batch_size}_lr_{learning_rate}.sh'
    
    # 2. 条件逻辑：如果 shuffle_train_dataset 为 True，则 shuffle_flag 为参数字符串，否则为空字符串
    shuffle_flag = "--shuffle_train_dataset \\" if shuffle_train_dataset else ""

    # 使用 f-string 构建 shell 脚本内容
    script_content = f"""#!/bin/bash
cd
source /volume/pt-train/users/wzhang/ghchen/laip/miniconda3/etc/profile.d/conda.sh

cd /volume/pt-train/users/wzhang/ghchen/laip/code/OpenRLHF
conda activate openrlhf

export WANDB_MODE=offline

set -x

# 使用 heredoc 定义多行训练命令，以提高可读性
read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \\
   --save_path {save_path} \\
   --save_steps -1 \\
   --logging_steps 1 \\
   --eval_steps -1 \\
   --lr_warmup_ratio 0.1 \\
   --train_batch_size {train_batch_size} \\
   --micro_train_batch_size {micro_train_batch_size} \\
   --pretrain {model_path} \\
   --bf16 \\
   {shuffle_flag}
   --max_epochs {max_epochs} \\
   --max_len 2048 \\
   --zero_stage 3 \\
   --learning_rate {learning_rate} \\
   --beta {beta} \\
   --dataset {dataset_path} \\
   --dataset_split train \\
   --apply_chat_template \\
   --chosen_key chosen \\
   --rejected_key rejected \\
   --flash_attn \\
   --load_checkpoint \\
   --gradient_checkpointing \\
   --seed 42 \\
   --wandb_run_name {save_path.split('/')[-1]} \\
   --use_wandb 8ab092da28de79c21207a141b3f6f683cca27932
EOF

if [[ "${{1}}" != "slurm" ]]; then
    deepspeed --module $training_commands
else
    echo "检测到 Slurm 执行模式。脚本将不会直接运行 deepspeed 命令。"
    echo "将要执行的训练指令如下:"
    echo "$training_commands"
fi
"""
    # 过滤掉由于条件插入可能产生的空行
    script_content = os.linesep.join([s for s in script_content.splitlines() if s.strip()])


    # 确保脚本的输出目录存在
    if sh_path:
        os.makedirs(os.path.dirname(sh_path), exist_ok=True)

    # 将生成的脚本内容写入文件
    with open(sh_path, 'w') as f:
        f.write(script_content)

    # 向用户打印确认信息
    print(f"✔ 训练检查点将保存至: {save_path}")
    print(f"✔ 可执行的 shell 脚本已生成: {sh_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="为 OpenRLHF DPO 训练生成一个 DeepSpeed 启动脚本。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 定义命令行参数
    parser.add_argument("--model_path", type=str, required=True, help="用于DPO训练的预训练SFT模型路径。")
    parser.add_argument("--dataset_path", type=str, required=True, help="偏好数据集的名称或路径。")
    parser.add_argument("--sh_path", type=str, default=None, help="[可选] 生成的 .sh 文件的保存路径。")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="学习率。")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO 的 β 参数。")
    parser.add_argument("--max_epochs", type=int, default=1, help="训练轮数。")
    parser.add_argument("--train_batch_size", type=int, default=128, help="全局批次大小。")
    parser.add_argument("--micro_train_batch_size", type=int, default=1, help="单设备批次大小。")
    parser.add_argument("--shuffle_train_dataset", action='store_true', help="如果提供此标志，则会对训练数据集进行 shuffle。")

    args = parser.parse_args()

    create_dpo_script(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        sh_path=args.sh_path,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        micro_train_batch_size=args.micro_train_batch_size,
        shuffle_train_dataset=args.shuffle_train_dataset # 将解析出的值传递给函数
    )
