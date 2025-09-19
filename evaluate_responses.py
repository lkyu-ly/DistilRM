#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import random
import os
import json

# 路径设置
DATA_PATH = "data/skywork_10k.jsonl"
RESPONSE_DIR = "responses"
OUTPUT_DIR = "evaluations"
TEACHER_MODEL_PATH = "models/Qwen3-14B"
CANDIDATE_MODELS = [
    "llama-3.1-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "gemma-2-9b-it",
    "vicuna-7b-v1.5"
]

# 评估 Prompt 模板
EVAL_PROMPT_TEMPLATE = """[Question]
{question}

[The Start of Assistant 1's Answer]
{answer_1}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer_2}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

### Response:"""


def load_prompts(file_path):
    """加载 skywork_10k.jsonl 的 prompts"""
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


def load_candidate_responses():
    """加载所有候选模型的 responses"""
    responses_by_model = {}
    for model_name in CANDIDATE_MODELS:
        file_path = os.path.join(
            RESPONSE_DIR, f"responses_{model_name.replace('/', '_')}.jsonl")
        responses = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        responses.append(data)
                    except json.JSONDecodeError as e:
                        print(f"错误：无法解析 {file_path} 的行：{line.strip()}，错误：{e}")
            responses_by_model[model_name] = responses
            print(f"成功加载 {model_name} 的 {len(responses)} 个 responses")
        except Exception as e:
            print(f"错误：无法读取 {file_path}：{e}")
    return responses_by_model


def sample_two_responses(responses_by_model, question):
    """为给定 question 随机采样两个候选模型的 response"""
    available_models = [m for m in CANDIDATE_MODELS if any(
        r['question'] == question for r in responses_by_model[m])]
    if len(available_models) < 2:
        print(f"警告：{question} 缺少足够候选模型的 response，跳过")
        return None
    selected_models = random.sample(available_models, 2)
    response_1 = next(
        r for r in responses_by_model[selected_models[0]] if r['question'] == question)
    response_2 = next(
        r for r in responses_by_model[selected_models[1]] if r['question'] == question)
    return {
        "assistant_1_model": selected_models[0],
        "assistant_1_response": response_1['response'],
        "assistant_2_model": selected_models[1],
        "assistant_2_response": response_2['response']
    }


def format_eval_prompt(question, answer_1, answer_2):
    """格式化评估 prompt"""
    return EVAL_PROMPT_TEMPLATE.format(
        question=question,
        answer_1=answer_1,
        answer_2=answer_2
    )


def parse_teacher_output(output):
    """解析教师模型输出，提取评分和说明"""
    try:
        lines = output.strip().split('\n')
        if not lines:
            return None, None
        # 提取第一行的两个分数
        score_line = lines[0].strip()
        scores = score_line.split()
        if len(scores) != 2 or not all(s.isdigit() and 1 <= int(s) <= 10 for s in scores):
            print(f"警告：无效评分格式：{score_line}")
            return None, None
        return f"{scores[0]} {scores[1]}"
    except Exception as e:
        print(f"错误：无法解析教师输出：{output}，错误：{e}")
        return None, None


def evaluate_responses(teacher_model_path, prompts, responses_by_model):
    """使用 Qwen3-14B 评估候选模型 responses"""
    try:
        # 加载教师模型和 tokenizer
        print("加载教师模型 Qwen3-14B...")
        llm = LLM(model=teacher_model_path,
                  trust_remote_code=True, max_model_len=8192)
        tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_path, trust_remote_code=True)

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10
        )

        # 准备评估 prompts
        eval_prompts = []
        eval_metadata = []
        for question in tqdm(prompts, desc="准备评估 prompts"):
            sample = sample_two_responses(responses_by_model, question)
            if not sample:
                continue
            eval_prompt_text = format_eval_prompt(
                question,
                sample['assistant_1_response'],
                sample['assistant_2_response']
            )
            # 转为对话格式
            messages = [
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": eval_prompt_text}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            eval_prompts.append(formatted_prompt)
            eval_metadata.append({
                "question": question,
                "assistant_1_model": sample['assistant_1_model'],
                "assistant_1_response": sample['assistant_1_response'],
                "assistant_2_model": sample['assistant_2_model'],
                "assistant_2_response": sample['assistant_2_response']
            })

        # 批量推理
        print("开始教师模型推理...")
        outputs = llm.generate(eval_prompts, sampling_params)

        # 保存结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, "evaluations.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, output in enumerate(tqdm(outputs, desc="处理教师输出")):
                scores = parse_teacher_output(
                    output.outputs[0].text)
                if scores is None:
                    continue
                result = {
                    "question": eval_metadata[i]["question"],
                    "assistant_1_model": eval_metadata[i]["assistant_1_model"],
                    "assistant_1_response": eval_metadata[i]["assistant_1_response"],
                    "assistant_2_model": eval_metadata[i]["assistant_2_model"],
                    "assistant_2_response": eval_metadata[i]["assistant_2_response"],
                    "scores": scores
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"已保存评价结果到 {output_file}")

    except Exception as e:
        print(f"错误：教师模型推理失败：{e}")


def main():
    random.seed(42)
    # 加载 prompts 和 responses
    prompts = load_prompts(DATA_PATH)
    if not prompts:
        print("错误：没有加载到任何 prompts，退出")
        return
    responses_by_model = load_candidate_responses()
    if not responses_by_model:
        print("错误：没有加载到任何 responses，退出")
        return

    # 运行评估
    evaluate_responses(TEACHER_MODEL_PATH, prompts, responses_by_model)


if __name__ == "__main__":
    main()
