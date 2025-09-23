from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)


def build_joint_dataset(
    data_path,
    tokenizer,
    teacher_tokenizer,
    split="train",
    size=None,
    model_name="",
    max_length=512,
):
    try:
        ds = load_dataset("json", data_files=data_path, split="train")
        logging.info(f"Loaded joint dataset from {data_path} with {len(ds)} examples")
    except Exception as e:
        logging.error(f"Failed to load dataset from {data_path}: {e}")
        raise

    if size is not None:
        ds = ds.select(range(min(size, len(ds))))
        logging.info(f"Reduced dataset to {len(ds)} examples")

    def convert_to_chat_format(example):
        prompt = example.get("prompt", "").strip()
        chosen = example.get("chosen", "").strip() or " "
        rejected = example.get("rejected", "").strip() or " "
        teacher_response = example.get("teacher_response", "").strip() or " "

        if not prompt:
            raise ValueError(f"Empty prompt in example: {example}")

        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ]
        teacher_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": teacher_response},
        ]

        result = {
            "chosen": chosen_messages,
            "rejected": rejected_messages,
            # "teacher_response": teacher_response, ???
            "teacher_response": teacher_messages,
        }
        chosen_score, rejected_score = example.get("chosen_score"), example.get(
            "rejected_score"
        )
        if chosen_score is not None:
            result["chosen_score"] = chosen_score
        if rejected_score is not None:
            result["rejected_score"] = rejected_score

        return result

    def formatting_func(example):
        try:
            example = convert_to_chat_format(example)
            kwargs = {
                "padding": "max_length",
                "truncation": True,
                "max_length": max_length,
                "return_tensors": "pt",
            }

            # Tokenize chosen and rejected for RM training
            prompt_plus_chosen_response = tokenizer.apply_chat_template(
                example["chosen"], tokenize=False
            )
            prompt_plus_rejected_response = tokenizer.apply_chat_template(
                example["rejected"], tokenize=False
            )

            if (
                not prompt_plus_chosen_response.strip()
                or not prompt_plus_rejected_response.strip()
            ):
                logging.warning(f"Empty tokenized response in example: {example}")
                return None

            tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
            tokens_rejected = tokenizer.encode_plus(
                prompt_plus_rejected_response, **kwargs
            )

            for key, tokens in [
                ("chosen", tokens_chosen),
                ("rejected", tokens_rejected),
            ]:
                if (
                    tokens["input_ids"].shape[1] != max_length
                    or tokens["attention_mask"].shape[1] != max_length
                ):
                    logging.warning(f"Shape mismatch in {key} for example: {example}")
                    return None

            # Create labels for SFT
            prompt = example["chosen"][:-1]
            prompt_template = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)[
                "input_ids"
            ][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_chosen[: len(tokens_prompt)] = -100
            label_rejected[: len(tokens_prompt)] = -100

            # Tokenize teacher response for KL distillation
            teacher_kwargs = kwargs.copy()
            if teacher_tokenizer:
                prompt_plus_teacher_response = teacher_tokenizer.apply_chat_template(
                    example["teacher_response"], tokenize=False
                )
                teacher_tokens = teacher_tokenizer.encode_plus(
                    prompt_plus_teacher_response, **teacher_kwargs
                )

                # Get teacher prefix length
                teacher_prompt = example["teacher_response"][:-1]
                teacher_prompt_template = teacher_tokenizer.apply_chat_template(
                    teacher_prompt, tokenize=False, add_generation_prompt=True
                )
                teacher_tokens_prompt = teacher_tokenizer.encode_plus(
                    teacher_prompt_template, **teacher_kwargs
                )["input_ids"][0]
                teacher_prefix_len = len(teacher_tokens_prompt)
            else:
                # If no teacher tokenizer, use student tokenizer
                teacher_tokens = tokens_chosen
                teacher_prefix_len = len(tokens_prompt)

            student_prefix_len = len(tokens_prompt)

            return {
                # RM training data
                "input_ids_chosen": tokens_chosen["input_ids"][0],
                "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0],
                "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,
                "label_rejected": label_rejected,
                # KL distillation data
                "teacher_input_ids": teacher_tokens["input_ids"][0],
                "teacher_attention_mask": teacher_tokens["attention_mask"][0],
                "teacher_prefix_len": teacher_prefix_len,
                "student_prefix_len": student_prefix_len,
            }
        except Exception as e:
            logging.error(f"Error processing example {example}: {e}")
            raise

    ds = ds.map(formatting_func, batched=False, num_proc=16)
    ds = ds.filter(lambda x: x is not None)
    logging.info(f"Joint dataset tokenized, columns: {ds.column_names}")

    def validate_tokens(example):
        # Validate RM data
        rm_keys = [
            ("input_ids_chosen", "attention_mask_chosen"),
            ("input_ids_rejected", "attention_mask_rejected"),
        ]
        for input_key, mask_key in rm_keys:
            input_ids = example[input_key]
            attention_mask = example[mask_key]
            if len(input_ids) != max_length or len(attention_mask) != max_length:
                logging.warning(
                    f"Length mismatch for {input_key} in example: {example}"
                )
                return False
            valid_token_count = sum(attention_mask)
            if valid_token_count <= 1:
                logging.warning(
                    f"Too few valid tokens in {input_key} for example: {example}, count: {valid_token_count}"
                )
                return False

        # Validate teacher data
        teacher_keys = [("teacher_input_ids", "teacher_attention_mask")]
        for input_key, mask_key in teacher_keys:
            input_ids = example[input_key]
            attention_mask = example[mask_key]
            if len(input_ids) != max_length or len(attention_mask) != max_length:
                logging.warning(
                    f"Length mismatch for {input_key} in example: {example}"
                )
                return False

        # Validate prefix lengths
        if not isinstance(example["teacher_prefix_len"], int) or not isinstance(
            example["student_prefix_len"], int
        ):
            logging.warning(f"Invalid prefix lengths in example: {example}")
            return False

        return True

    ds = ds.filter(validate_tokens)
    logging.info(f"Joint dataset after validation: {len(ds)} examples")

    remove_columns = [
        col
        for col in ds.column_names
        if not any(
            key in col
            for key in [
                "input_ids",
                "attention_mask",
                "label",
                "teacher",
                "student_prefix",
            ]
        )
    ]
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


def load_joint_train_eval_dataset(
    data_path,
    tokenizer,
    teacher_tokenizer,
    size=None,
    mode="",
    model_name="",
    max_length=512,
):
    dataset = build_joint_dataset(
        data_path,
        tokenizer,
        teacher_tokenizer,
        split="train",
        size=size,
        model_name=model_name,
        max_length=max_length,
    )
    dataset_split = dataset.train_test_split(test_size=0.01)
    train_dataset, eval_dataset = dataset_split["train"], dataset_split["test"]
    logging.info(
        f"Joint train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}"
    )
    return train_dataset, eval_dataset
