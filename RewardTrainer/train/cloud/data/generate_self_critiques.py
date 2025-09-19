import os
from argparse import ArgumentParser

from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from train.cloud.train.train import COT_PROMPT
from train.cloud.train.data import build_chat_messages

def build_feedback_prompts(tokenizer, example):
    if "qwen" in tokenizer.name_or_path.lower():
        bos_text = "<|im_start|>"
    else:
        bos_text = tokenizer.decode([tokenizer.bos_token_id])
    eos_text = tokenizer.decode([tokenizer.eos_token_id])
    
    # 根据模型设置 eot_text
    if "llama-3" in tokenizer.name_or_path.lower():
        eot_text = "<|eot_id|>"
    elif "mistral" in tokenizer.name_or_path.lower():
        eot_text = "[/INST]"
    elif "qwen" in tokenizer.name_or_path.lower():
        eot_text = "</s>"

    chosen_prefix = tokenizer.apply_chat_template(build_chat_messages(example["prompt"], example["chosen"]), tokenize=False)
    rejected_prefix = tokenizer.apply_chat_template(build_chat_messages(example["prompt"], example["rejected"]), tokenize=False)
    cot_fmt = tokenizer.apply_chat_template([{"role": "user", "content": COT_PROMPT}], tokenize=False).replace(bos_text, "").replace(eos_text, "").replace(eot_text, "")

    example["chosen_feedback_prompt"] = chosen_prefix + cot_fmt
    example["rejected_feedback_prompt"] = rejected_prefix + cot_fmt
    return example

def main(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize vLLM model
    llm = LLM(model=args.model, max_model_len=args.max_tokens, gpu_memory_utilization=0.6)

    # 根据模型是否为 llama3 设置 eot_text
    if "llama-3" in tokenizer.name_or_path.lower():
        eot_text = "<|eot_id|>"
    elif "mistral" in tokenizer.name_or_path.lower():
        eot_text = "[/INST]"
    elif "qwen" in tokenizer.name_or_path.lower():
        eot_text = "<|im_end|>"

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temp,
        max_tokens=args.max_tokens,
        stop=[eot_text, tokenizer.eos_token]  # Stop tokens
    )

    # Load the dataset from local file
    file_extension = os.path.splitext(args.base_dataset)[1].lower()
    if file_extension == ".json" or file_extension == ".jsonl":
        dataset_format = "json"
    elif file_extension == ".csv":
        dataset_format = "csv"
    elif file_extension == ".parquet":
        dataset_format = "parquet"
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats are .json, .jsonl, .csv, .parquet")

    # Load dataset from local file
    ds = load_dataset(dataset_format, data_files=args.base_dataset, split="train")

    ds = ds.map(lambda x: build_feedback_prompts(tokenizer, x), num_proc=args.num_proc)

    chosen_feedback_prompts = [example["chosen_feedback_prompt"] for example in ds]
    rejected_feedback_prompts = [example["rejected_feedback_prompt"] for example in ds]

    # Generate chosen feedback
    chosen_outputs = llm.generate(chosen_feedback_prompts, sampling_params)
    chosen_feedback = [output.outputs[0].text for output in chosen_outputs]

    # Generate rejected feedback
    rejected_outputs = llm.generate(rejected_feedback_prompts, sampling_params)
    rejected_feedback = [output.outputs[0].text for output in rejected_outputs]

    all_feedback = []
    for i in tqdm(range(len(ds)), desc="Processing examples"):
        all_feedback.append({
            **ds[i],
            "chosen_feedback": [chosen_feedback[i]],
            "rejected_feedback": [rejected_feedback[i]]
        })

    hf_feedback_ds = Dataset.from_list(all_feedback)
    cols_to_select = ["prompt", "chosen", "rejected", "chosen_feedback", "rejected_feedback", "id"]
    hf_feedback_ds = hf_feedback_ds.select_columns(cols_to_select)

    # Save to local JSON file as a single array
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(hf_feedback_ds.to_list(), f, indent=2, ensure_ascii=False)
    print(f"Dataset saved to {args.output}")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Model / data params
    parser.add_argument("--model", type=str, required=True, help="Path to the model or model identifier (e.g., meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--base-dataset", type=str, required=True, help="Path to local dataset file (e.g., .json, .jsonl, .csv, .parquet)")
    parser.add_argument("--output", type=str, default="output.json", help="Path to output JSON file")
    parser.add_argument("--num-proc", type=int, default=10, help="Number of processes for dataset mapping")

    # Sampling params
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")

    args = parser.parse_args()

    main(args)
