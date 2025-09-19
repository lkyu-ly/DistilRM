#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# 输入文件路径
DATA_PATH = "data/skywork_10k.jsonl"
# 输出目录
OUTPUT_DIR = "responses"
# 候选模型列表
MODEL_PATHS = {
    # "llama-3.1-8b-instruct": "models/Llama-3.1-8B-Instruct",
    # "mistral-7b-instruct-v0.3": "models/Mistral-7B-Instruct-v0.3",
    # "gemma-2-9b-it": "models/gemma-2-9b-it",
    # "vicuna-7b-v1.5": "models/vicuna-7b-v1.5",
    # "qwen3-14b": "models/Qwen3-14B",
    "qwen2.5-3B-instruct_qwen3-14b": "/root/autodl-tmp/output//kl_distill_Qwen2.5-3B-Instruct_Qwen3-14B",
    # "qwen2.5-3B-instruct": "models/Qwen2.5-3B-Instruct",
}

def load_prompts(file_path):
    """从 JSONL 文件加载 prompts"""
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'question' in data:
                        prompts.append(data['question'])
                    else:
                        print(f"警告：跳过无效行，缺少 'question' 字段：{line.strip()}")
                except json.JSONDecodeError as e:
                    print(f"错误：无法解析 JSON 行：{line.strip()}，错误：{e}")
        print(f"成功加载 {len(prompts)} 个 prompts")
        return prompts
    except Exception as e:
        print(f"错误：无法读取文件 {file_path}：{e}")
        return []

def format_prompt(prompt, tokenizer, model_name):
    """格式化提示为模型可接受的对话格式"""
    messages = [{"role": "user", "content": prompt}]
    try:
        # 使用 tokenizer 的 chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return formatted_prompt
    except Exception as e:
        print(f"警告：{model_name} 的 chat template 失败：{e}")
        raise ValueError(f"无法格式化提示：{prompt}")

def generate_responses(model_path, model_name, prompts):
    """使用 vLLM 推理生成 responses"""
    try:
        # 加载模型和 tokenizer
        print(f"加载模型：{model_name}")
        llm = LLM(model=model_path, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.5)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 测试第一条 prompt 的模板处理
        test_prompt = prompts[0]
        print(f"\n原始 prompt:\n{test_prompt}")

        formatted_prompt = format_prompt(test_prompt, tokenizer, model_name)
        print(f"\n格式化后的 prompt:\n{formatted_prompt}")
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024
        )

        # 格式化 prompts
        formatted_prompts = [format_prompt(prompt, tokenizer, model_name) for prompt in prompts]

        # 批量推理
        print(f"开始为 {model_name} 生成 responses...")
        outputs = llm.generate(formatted_prompts, sampling_params)

        # 收集结果
        responses = []
        for i, output in enumerate(tqdm(outputs, desc=f"处理 {model_name} 输出")):
            response = output.outputs[0].text.strip()
            responses.append({
                "question": prompts[i],
                "model": model_name,
                "response": response
            })

        # 保存结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"responses_{model_name.replace('/', '_')}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for response in responses:
                f.write(json.dumps(response, ensure_ascii=False) + '\n')
        print(f"已保存 {model_name} 的 responses 到 {output_file}")

        return responses

    except Exception as e:
        print(f"错误：{model_name} 推理失败：{e}")
        return []

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 prompts
    prompts = load_prompts(DATA_PATH)
    if not prompts:
        print("错误：没有加载到任何 prompts，退出")
        return

    # 为每个模型生成 responses
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"警告：模型路径 {model_path} 不存在，跳过 {model_name}")
            continue
        generate_responses(model_path, model_name, prompts)

if __name__ == "__main__":
    main()