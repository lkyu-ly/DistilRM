import json
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm   # ✅ 新加的

# 教师模型
MODEL_NAME = "/root/autodl-tmp/models/Qwen3-14B"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16
)


SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Math": ["math-prm"],
    "Code": [
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ]
}


def compute_logprobs_batch(prompt, responses):
    """
    批量计算 log-prob，返回每个 response 的平均 log-prob
    """
    batch_texts = []
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    prompt_len = len(tokenizer(prompt_text)["input_ids"])

    for resp in responses:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        batch_texts.append(full_text)

    encodings = tokenizer(batch_texts, return_tensors="pt",
                          padding=True).to(device)

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits

    log_probs = []
    for i, resp in enumerate(responses):
        input_ids = encodings["input_ids"][i]
        attention_mask = encodings["attention_mask"][i]

        # 计算 response 部分的起始位置
        full_len = attention_mask.sum().item()
        resp_start = prompt_len
        resp_ids = input_ids[resp_start:full_len]

        # 对应预测 logits
        logits_slice = logits[i, resp_start - 1: full_len - 1]
        log_softmax = torch.log_softmax(logits_slice, dim=-1)

        token_logps = log_softmax[torch.arange(len(resp_ids)), resp_ids]
        log_probs.append(token_logps.mean().item())

    return log_probs


def evaluate(file_path):
    results = defaultdict(lambda: {"correct": 0, "total": 0})

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Evaluating"):
        data = json.loads(line)
        prompt = data["prompt"]
        chosen = data["chosen"]
        rejected = data["rejected"]
        subset = data["subset"]

        chosen_lp, rejected_lp = compute_logprobs_batch(
            prompt, [chosen, rejected])

        if chosen_lp > rejected_lp:
            results[subset]["correct"] += 1
        results[subset]["total"] += 1

    # 统计大类准确率
    accuracies = {}
    for cat, subsets in SUBSET_MAPPING.items():
        correct = sum(results[s]["correct"] for s in subsets if s in results)
        total = sum(results[s]["total"] for s in subsets if s in results)
        accuracies[cat] = correct / total if total > 0 else 0.0

    # Reasoning = Math 和 Code 的平均
    reasoning_acc = (accuracies["Math"] + accuracies["Code"]) / 2
    accuracies["Reasoning"] = reasoning_acc

    # Overall = Chat, Chat Hard, Safety, Reasoning 的平均
    overall_acc = (accuracies["Chat"] + accuracies["Chat Hard"] +
                   accuracies["Safety"] + reasoning_acc) / 4
    accuracies["Overall"] = overall_acc

    return accuracies


if __name__ == "__main__":
    file_path = "/home/DistilRM/RewardTrainer/data/rewardbench/filtered.json"
    accs = evaluate(file_path)
    print("Chat Acc:", accs["Chat"])
    print("Chat Hard Acc:", accs["Chat Hard"])
    print("Safety Acc:", accs["Safety"])
    print("Reasoning Acc:", accs["Reasoning"])
    print("Overall Acc:", accs["Overall"])
