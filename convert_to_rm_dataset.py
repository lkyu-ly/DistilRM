#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import random
import uuid
from tqdm import tqdm

# 路径设置
EVAL_PATH = "evaluations/evaluations.jsonl"
OUTPUT_PATH = "RewardTrainer/data/skywork_10k_rm.json"


def load_evaluations(file_path):
    """加载 evaluations.jsonl"""
    evaluations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    evaluations.append(data)
                except json.JSONDecodeError as e:
                    print(f"错误：无法解析 JSON 行：{line.strip()}，错误：{e}")
        print(f"成功加载 {len(evaluations)} 个评价记录")
        return evaluations
    except Exception as e:
        print(f"错误：无法读取文件 {file_path}：{e}")
        return []


def convert_to_rm_format(evaluations):
    """将 evaluations 转换为 RM 训练数据集格式"""
    rm_dataset = []
    skipped = 0
    for eval_data in tqdm(evaluations, desc="转换评价数据"):
        # 检查必要字段
        required_fields = ["question", "assistant_1_response",
                           "assistant_2_response", "scores"]
        if not all(field in eval_data for field in required_fields):
            print(f"警告：缺少必要字段，跳过：{eval_data.get('question', '')[:50]}...")
            skipped += 1
            continue

        # 解析评分
        try:
            score_1, score_2 = map(int, eval_data["scores"].split())
            if not (1 <= score_1 <= 10 and 1 <= score_2 <= 10):
                print(
                    f"警告：无效评分 {eval_data['scores']}，跳过：{eval_data['question'][:50]}...")
                skipped += 1
                continue
        except ValueError as e:
            print(
                f"错误：无法解析评分 {eval_data['scores']}，跳过：{eval_data['question'][:50]}...，错误：{e}")
            skipped += 1
            continue

        # 确定 chosen 和 rejected
        if score_1 > score_2:
            chosen = eval_data["assistant_1_response"]
            rejected = eval_data["assistant_2_response"]
            chosen_score = score_1
            rejected_score = score_2
        elif score_2 > score_1:
            chosen = eval_data["assistant_2_response"]
            rejected = eval_data["assistant_1_response"]
            chosen_score = score_2
            rejected_score = score_1
        else:
            skipped += 1
            continue

        # 生成 RM 数据点
        rm_data = {
            "id": str(uuid.uuid4()),
            "prompt": eval_data["question"],
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score
        }
        rm_dataset.append(rm_data)

    print(f"转换完成：生成 {len(rm_dataset)} 个数据点，跳过 {skipped} 个无效记录")
    return rm_dataset


def save_rm_dataset(rm_dataset, output_path):
    """保存 RM 数据集为 JSON 文件"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rm_dataset, f, ensure_ascii=False, indent=2)
        print(f"已保存 RM 数据集到 {output_path}")
    except Exception as e:
        print(f"错误：无法保存文件 {output_path}：{e}")


def main():
    # 加载评价数据
    evaluations = load_evaluations(EVAL_PATH)
    if not evaluations:
        print("错误：没有加载到任何评价数据，退出")
        return

    # 转换为 RM 格式
    rm_dataset = convert_to_rm_format(evaluations)
    if not rm_dataset:
        print("错误：没有生成任何 RM 数据点，退出")
        return

    # 保存结果
    save_rm_dataset(rm_dataset, OUTPUT_PATH)


if __name__ == "__main__":
    main()
