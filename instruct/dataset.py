from torch.utils.data import Dataset
import json
from typing import Any
from transformers import LlamaTokenizerFast
import torch
import random

import os

pretrained_path = f"{os.path.dirname(__file__)}/weights/"

tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def preprocess(examples: dict[str, Any], max_length: int) -> dict[str, Any]:
    results = {}

    context = examples["story"]
    questions = examples["questions"]
    answers = examples["answers"]

    tokenized_inputs = [
        tokenizer.encode_plus(
            f"Question: {question['input_text']}, Context:{context}",
            return_tensors="pt",
            add_special_tokens = False
        )
        for question in questions
    ]
    combined_qct = [
        f"Question: {question['input_text']}, Context:{context}, Answer: {answer['input_text']}"
        for question, answer in zip(questions, answers)
    ]
    combined_inputs = tokenizer.batch_encode_plus(
        combined_qct,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
        return_overflowing_tokens=False,
        add_special_tokens = False
    )
    input_ids = combined_inputs["input_ids"]
    attention_mask = combined_inputs["attention_mask"]

    labels = input_ids.clone().detach()
    labels[labels == tokenizer.pad_token_id] = -100

    for idx, tokenized_input in enumerate(tokenized_inputs):
        input_ids = tokenized_input["input_ids"][0]
        labels[idx, : len(input_ids) + 1] = -100

    results["input_ids"] = combined_inputs["input_ids"][0]
    results["attention_mask"] = combined_inputs["attention_mask"][0]
    results["labels"] = labels[0]

    return results


class InstructDataset(Dataset):
    def __init__(
        self,
        json_file,
        max_length=2048,
        dataset_size=None,
    ):
        with open(json_file) as fd:
            self.dataset = json.load(fd)["data"]

        if dataset_size is not None:
            self.dataset = self.dataset[:dataset_size]
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        example = self.dataset[index]
        processed_examples = preprocess(example, self.max_length)
        return processed_examples


if __name__ == "__main__":
    dataset = InstructDataset(
        json_file="/home/chief/instruction-tuning/instruct/dataset/coqa-train-v1.0.json",
    )
    print(dataset[3])
