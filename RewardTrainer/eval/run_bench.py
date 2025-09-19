import json
import os
import argparse

from omegaconf import OmegaConf
from datasets import load_dataset

from module import InferenceModule, VllmModule, HfModule, OpenaiModule


BENCHMARK_IDS = ["rewardbench"]


def make_data_row(id: int, instruction: str, response1: str, response2: str, label: int) -> dict:
    return {
        "id": id,
        "instruction": instruction.strip(),
        "response1": response1.strip(),
        "response2": response2.strip(),
        "label": label,
    }


def get_benchmark_data(benchmark_id: str, data_path) -> dict:
    """output a standardized dataset. only the contents.
    the data structure will be kept until the results."""
    benchmark_set = {}
    assert benchmark_id in BENCHMARK_IDS
    print("Loading benchmark:", benchmark_id)

    if benchmark_id == "rewardbench":
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
        dataset = []
        with open(os.path.join(data_path, "rewardbench/filtered.json"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
        for subset_name in ["Chat", "Chat Hard", "Safety", "Math", "Code"]:
            subset = []
            for i, row in enumerate(dataset):
                if dataset[i]["subset"] in SUBSET_MAPPING[subset_name]:
                    subset.append(make_data_row(
                        i, row["prompt"], row["chosen"], row["rejected"], 1))
            benchmark_set[subset_name] = subset
    else:
        raise ValueError(benchmark_id)

    return benchmark_set


def add_inference(benchmark_data: dict, module: InferenceModule) -> None:
    """all common logic for benchmarking. 
    apply swap, apply prompt template, apply chat template, for all subsets in benchmark data.
    run inference and update on benchmark_data"""
    conversation_list = []

    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            for swap in [False, True]:
                conversation_list.append(module.make_conversation(
                    row["instruction"], row["response1"], row["response2"], swap))

    generated_texts = module.generate(conversation_list)

    index = 0
    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            result = {}
            for swap_id in ["orig", "swap"]:
                result[swap_id] = {"completion": generated_texts[index]}
                index += 1
            row["result"] = result
    assert (len(generated_texts) == index)


def add_parse_result(benchmark_data: dict, module: InferenceModule) -> None:
    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            for swap, swap_id in [(False, "orig"), (True, "swap")]:
                result_dict = row["result"][swap_id]
                completion = result_dict["completion"]
                result_dict["prediction"] = module.get_prediction(completion)
                result_dict["is_correct"] = module.is_correct(
                    result_dict["prediction"], row["label"], swap)


def get_model_statistics(run_name: str) -> dict:
    """read all inference results for the model and return scores"""
    model_stats = {}
    for benchmark_id in BENCHMARK_IDS:
        benchmark_result = {}
        filename = f"result/{run_name}/{benchmark_id}.json"
        if not os.path.exists(filename):
            print("result file", filename, "does not exist.")
            continue
        with open(filename) as f:
            data = json.load(f)
        for subset_name, subset in data.items():
            stats = {key: 0 for key in ["single_total", "single_correct", "single_accuracy",
                                        "pair_total", "pair_correct", "pair_accuracy", "pair_agree", "pair_agreement_rate"]}
            for row in subset:
                stats["single_total"] += 2
                stats["pair_total"] += 1
                if row["result"]["orig"]["is_correct"]:
                    stats["single_correct"] += 1
                if row["result"]["swap"]["is_correct"]:
                    stats["single_correct"] += 1
                if row["result"]["orig"]["is_correct"] and row["result"]["swap"]["is_correct"]:
                    stats["pair_correct"] += 1
                pred_orig = row["result"]["orig"]["prediction"]
                pred_swap = row["result"]["swap"]["prediction"]
                if set([pred_orig, pred_swap]) in [set([1, 2]), set([3])]:
                    stats["pair_agree"] += 1

            stats["single_accuracy"] = round(
                stats["single_correct"] / stats["single_total"]*100, 2)
            stats["pair_accuracy"] = round(
                stats["pair_correct"] / stats["pair_total"]*100, 2)
            stats["pair_agreement_rate"] = round(
                stats["pair_agree"] / stats["pair_total"]*100, 2)
            benchmark_result[subset_name] = stats
        model_stats[benchmark_id] = benchmark_result
    return model_stats


def write_model_score(run_name: str) -> None:
    """create model's score file"""
    model_stats = get_model_statistics(run_name)

    with open(f"result/{run_name}/score.json", "w") as f:
        json.dump(model_stats, fp=f, ensure_ascii=False, indent=4)


def run_benchmark(run_name: str, args: argparse.Namespace) -> None:
    """run inference, parse and score."""
    os.makedirs("result", exist_ok=True)
    os.makedirs(f"result/{run_name}", exist_ok=True)

    config = OmegaConf.load(args.config)
    OmegaConf.save(config, f"result/{run_name}/config.yaml")
    print(config)

    if (not args.hf) and (config.get("vllm_args")):
        module = VllmModule(config=config)
    elif (args.hf) and (config.get("hf_args")):
        module = HfModule(config=config)
    elif config.get("openai_args"):
        module = OpenaiModule(config=config)
    else:
        raise NotImplementedError

    for benchmark_id in args.benchmarks:
        benchmark_data = get_benchmark_data(benchmark_id, args.data_path)

        add_inference(benchmark_data, module)
        add_parse_result(benchmark_data, module)

        with open(f"result/{run_name}/{benchmark_id}.json", "w") as f:
            json.dump(benchmark_data, fp=f, ensure_ascii=False, indent=2)

    write_model_score(run_name)


def run_parse(run_name: str, args: argparse.Namespace) -> None:
    """redo parsing for existing inference results, and update score."""
    config = OmegaConf.load(args.config)
    print(config)

    module = InferenceModule(config=config)
    for benchmark_id in args.benchmarks:
        with open(f"result/{run_name}/{benchmark_id}.json") as f:
            benchmark_data = json.load(f)
        add_parse_result(benchmark_data, module)
        with open(f"result/{run_name}/{benchmark_id}.json", "w") as f:
            json.dump(benchmark_data, fp=f, ensure_ascii=False, indent=2)

    write_model_score(run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/llama2-7b.yaml")
    parser.add_argument(
        "--name", default="", help="run name of the inference. defaults to config name.")

    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument("--benchmarks", type=list_of_strings, default=["biasbench"],
                        help="to include all benchmarks, call as '--benchmarks llmbar,hhh,mtbench,biasbench'")
    parser.add_argument("--hf", action="store_true",
                        help="use hf generate instead of vllm")
    parser.add_argument("--parse", action="store_true",
                        help="no inference. just parse and score.")
    parser.add_argument("--score", action="store_true",
                        help="no inference. just score.")
    parser.add_argument("--data-path")

    args = parser.parse_args()
    print(args)

    run_name = os.path.basename(args.config).replace(".yaml", "")
    if args.hf:
        run_name += "_hf"
    if args.name:
        run_name = args.name
    print("Run name:", run_name)

    if args.score:
        write_model_score(run_name)
    elif args.parse:
        run_parse(run_name, args)
    else:
        run_benchmark(run_name, args)
