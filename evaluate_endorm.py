import json
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# 教师模型
MODEL_NAME = "models/Qwen2.5-3B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

GAMMA = 0.95  # 折扣因子
BETA = 0.0     # 最小权重
ALPHA = 1.0    # 温度系数

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


def compute_endogenous_reward_batch(prompt, responses):
    """
    计算每个 response 的内生奖励（Outcome Reward）。
    返回一个奖励值列表。
    """
    rewards = []

    # 先对 prompt 进行 tokenize，获取其长度
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    prompt_encoding = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = prompt_encoding["input_ids"].shape[1]

    for resp in responses:
        # 构造完整的对话历史
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

        # Tokenize 整个序列
        encoding = tokenizer(full_text, return_tensors="pt").to(device)
        input_ids = encoding["input_ids"][0]  # Shape: [seq_len]
        seq_len = input_ids.shape[0]

        # 我们只关心 response 部分，即从 prompt_len 开始到 seq_len-1
        # 注意：模型的 logits[i] 预测的是 token[i+1]
        resp_start_idx = prompt_len
        resp_end_idx = seq_len  # 最后一个 token 的下一个位置

        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

        # --- 计算每一步的即时奖励 ---
        step_rewards = []
        for h in range(resp_start_idx, resp_end_idx):
            # 当前 token: a_h = input_ids[h]
            # 预测当前 token 的 logits: logits[h-1] (因为logits[i]预测的是i+1)
            if h == 0:
                continue  # 第一个 token 没有前一个 logits 来预测它，跳过

            # 计算 log(π̂(aₕ|sₕ))
            logit_h_minus_1 = logits[h - 1]  # Shape: [vocab_size]
            log_probs_h = torch.log_softmax(logit_h_minus_1, dim=-1)
            log_prob_chosen = log_probs_h[input_ids[h]].item()

            # 计算 VQ̂(sₕ) = α * log(Σ exp(Q̂(sₕ, a) / α))
            # Q̂(sₕ, a) 就是 logits[h-1]
            v_s_h = ALPHA * \
                torch.logsumexp(logit_h_minus_1 / ALPHA, dim=-1).item()

            # 计算 VQ̂(sₕ₊₁)
            # sₕ₊₁ 是下一个状态，其 Q 值由 logits[h] 给出
            if h < resp_end_idx - 1:
                # 如果不是最后一个 token，正常计算下一个状态的 V 值
                v_s_h_plus_1 = ALPHA * \
                    torch.logsumexp(logits[h] / ALPHA, dim=-1).item()
            else:
                # 如果是最后一个 token，根据论文假设，V(s_{H+1}) = 0
                v_s_h_plus_1 = 0.0

            # 计算即时奖励: r(sₕ, aₕ) = α * log(π̂(aₕ|sₕ)) + VQ̂(sₕ) - VQ̂(sₕ₊₁)
            instant_reward = ALPHA * log_prob_chosen + v_s_h - v_s_h_plus_1

            # 应用折扣因子
            # h 是绝对位置，需要计算它在 response 中的相对位置
            relative_step = h - resp_start_idx + 1  # 从 1 开始计数
            discount_weight = max(GAMMA ** (relative_step - 1), BETA)

            # 加权后的即时奖励
            discounted_instant_reward = discount_weight * instant_reward
            step_rewards.append(discounted_instant_reward)

        # 计算整个 response 的总奖励 (Outcome Reward)
        total_reward = sum(step_rewards)
        rewards.append(total_reward)

    return rewards


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

        chosen_reward, rejected_reward = compute_endogenous_reward_batch(
            prompt, [chosen, rejected])

        if chosen_reward > rejected_reward:
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
    file_path = "RewardTrainer/data/rewardbench.json"
    accs = evaluate(file_path)
    print("\n=== Evaluation Results (Using Endogenous Reward) ===")
    print(f"Chat Acc:       {accs['Chat']:.4f}")
    print(f"Chat Hard Acc:  {accs['Chat Hard']:.4f}")
    print(f"Safety Acc:     {accs['Safety']:.4f}")
    print(f"Reasoning Acc:  {accs['Reasoning']:.4f}")
    print(f"Overall Acc:    {accs['Overall']:.4f}")
