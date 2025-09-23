#!/usr/bin/env python3
"""
把 responses_qwen3-14b.jsonl 里的 response 合并到 skywork_10k_rm.json 中，
匹配字段为 prompt（在 jsonl 里叫 question），
合并后的新字段命名为 teacher_response。
找不到匹配项的样本直接跳过。
结果写入 skywork_10k_joint.json
"""

import json
import jsonlines

# 1. 读入 skywork 原始数据
with open("./skywork_10k_rm.json", "r", encoding="utf-8") as f:
    skywork = json.load(f)  # 标准 json 列表

# 2. 读入 qwen3-14b 的 jsonl，并建立 prompt->response 映射
prompt2resp = {}
with jsonlines.open("./responses_qwen3-14b.jsonl") as reader:
    for obj in reader:
        prompt2resp[obj["question"]] = obj["response"]

# 3. 合并
merged = []
for item in skywork:
    prompt = item["prompt"]
    if prompt in prompt2resp:
        new_item = item.copy()
        new_item["teacher_response"] = prompt2resp[prompt]
        merged.append(new_item)

# 4. 写出
with open("./skywork_10k_joint.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"合并完成，共 {len(merged)} 条样本写入 skywork_10k_joint.json")
