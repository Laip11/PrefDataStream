from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Union, Dict, Any
import os


class unify_infer:

    def __init__(
        self,
        model_name_or_path: str,
        load_type: str = "vllm",
        device: str = "cuda",
        dtype: str = "bfloat16",
        **kwargs,
    ):
        from transformers import  AutoTokenizer
        self.model_name_or_path = model_name_or_path
        self.model_type = load_type  # vllm / hf
        assert load_type in ["vllm", "hf"]
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,trust_remote_code=True)

    # -----------------------------
    # Model Initialization
    # -----------------------------
    def _init_model(self, gpu_memory_utilization: float = 0.8):
        if self.model_type == "vllm":
            from vllm import LLM
            import torch

            self.model = LLM(
                model=self.model_name_or_path,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=self.dtype,
            )
        elif self.model_type == "hf":
            import torch
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map="auto",
                    torch_dtype=getattr(torch, self.dtype, torch.bfloat16),
                    trust_remote_code=True).eval()

    # -----------------------------
    # Load Chat Template
    # -----------------------------
    def _load_chat_template(self, customized_chat_template: str = None):

        if self.tokenizer.chat_template is not None:
            return

        # Try loading chat_template.jinja from model path
        jinja_path = os.path.join(self.model_name_or_path, "chat_template.jinja")

        if os.path.exists(jinja_path):
            with open(jinja_path, "r", encoding="utf-8") as f:
                self.tokenizer.chat_template = f.read()
            return

        # Use customized
        if customized_chat_template is not None:
            self.tokenizer.chat_template = customized_chat_template
            return

        raise ValueError("chat_template is missing. Provide via customized_chat_template.")

    # -----------------------------
    # Prompt Processing
    # -----------------------------
    def process_prompt(
        self,
        prompts: Union[str, List[str]],
        system_prompt: str = None,
        enable_thinking: bool = False,
        customized_chat_tempalte: str = None,
        padding_side: str = "left"
        ) -> List[Any]:
        from tqdm import tqdm

        if isinstance(prompts, str):
            prompts = [prompts]

        # Set pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load chat template
        self._load_chat_template(customized_chat_tempalte)

        self.tokenizer.padding_side = padding_side

        input_texts = []

        for prompt in tqdm(prompts):
            messages = [{"role": "user", "content": prompt}]
            if system_prompt is not None:
                messages.insert(0, {"role": "system", "content": system_prompt})

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            input_texts.append(text)

        return {'naive_prompt': prompts, 'input_texts':input_texts}

    # -----------------------------
    # Inference
    # -----------------------------
    def infer(
        self,
        input: Dict[list, Any],
        batch_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 1,
        repetition_penalty: float = 1.0,
        seed: int = 42,
        **kwargs,
    ) -> List[Dict]:
        results = []

        # -------- vLLM --------
        if self.model_type == "vllm":
            from vllm import SamplingParams

            sampling = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=seed,
                max_tokens=max_new_tokens,
                **kwargs,
            )
            input_text_ls = input['input_texts']
            outputs = self.model.generate(input_text_ls, sampling)
            responses = [o.outputs[0].text for o in outputs]

            for i, input_text in enumerate(input_text_ls):
                item = {
                    'naive_prompt': input['naive_prompt'][i],
                    "input_prompt": input_text,
                    "response": responses[i],
                }
                results.append(item)

            return results

        # -------- HF --------
        import torch
        from tqdm import trange
        for start in trange(0, len(input['input_texts']), batch_size):
            batch_naive_prompt = input['naive_prompt'][start : start + batch_size]
            batch_input_prompt = input['input_texts'][start : start + batch_size]
            batch_tensors = self.tokenizer(batch_input_prompt, return_tensors="pt", padding=True).to(self.model.device)

            generated = self.model.generate(
                **batch_tensors,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )

            responses = self.tokenizer.batch_decode(
                generated[:, batch_tensors['input_ids'].shape[-1] :],
                skip_special_tokens=True,
            )

            for idx, input_text in enumerate(batch_input_prompt):
                item = {
                    'naive_prompt': batch_naive_prompt[idx],
                    "input_prompt": input_text,
                    "response": responses[idx],
                }
                results.append(item)

        return results



# import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # -------------------------------
# # 1. 初始化统一推理类
# # -------------------------------
# infer_engine = unify_infer(
#     model_name_or_path="/data1/laip/models/Qwen/Qwen2.5-0.5B-Instruct",  # 你自己的模型目录或 HF 名称
#     load_type="hf",                                  # "vllm" / "hf"
#     dtype="bfloat16",
# )

# # 初始化模型
# infer_engine._init_model()

# # -------------------------------
# # 2. 六个 prompt
# # -------------------------------
# prompts = [
#     "Introduce yourself briefly.",
#     "What is the capital of France?",
#     "Explain what a transformer model is.",
#     "Write a short poem about winter.",
#     "Why is the sky blue?",
#     "Give me three tips for learning machine learning.",
# ]

# # 可选系统提示词
# system_prompt = "You are a helpful AI assistant."

# # -------------------------------
# # 3. 处理 prompt → 编码为输入张量
# # -------------------------------
# input_tensors = infer_engine.process_prompt(
#     prompts=prompts,
#     system_prompt=system_prompt,
#     enable_thinking=False,  # 如果你的模型支持 thinking，可以开
# )

# # -------------------------------
# # 4. 推理
# # -------------------------------
# results = infer_engine.infer(
#     input=input_tensors,
#     batch_size=4,            # 分两批跑
#     max_new_tokens=100,
#     temperature=0.7,
# )

# # -------------------------------
# # 5. 打印结果
# # -------------------------------
# print(results)