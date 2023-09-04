from torch.utils.data import Dataset
import json
from typing import Any
from transformers import LlamaTokenizerFast
import os

pretrained_path = f"{os.path.dirname(__file__)}/weights/"

tokenizer = LlamaTokenizerFast.from_pretrained(
    pretrained_path,
)

tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def preprocess(
    examples: dict[str, Any], question_idx: int, max_length: int
) -> dict[str, Any]:
    results = {}

    context = examples["story"]
    question = examples["questions"][question_idx]
    answer = examples["answers"][question_idx]

    tokenized_input = tokenizer.encode_plus(
        f"Question: {question['input_text']}, Context:{context}",
        return_tensors="pt",
        # add_special_tokens = False
    )
    combined_qca = f"Question: {question['input_text']}, Context:{context}, Answer: {answer['input_text']}"

    combined_input = tokenizer.encode_plus(
        combined_qca,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
        return_overflowing_tokens=False,
        # add_special_tokens = False
    )
    input_ids = combined_input["input_ids"]
    attention_mask = combined_input["attention_mask"]

    labels = input_ids.clone().detach()
    labels[labels == tokenizer.pad_token_id] = -100

    question_length = len(tokenized_input["input_ids"])
    labels[:, :question_length + 1] = -100

    shifted_labels = labels[:, 1:]  # shift labels to make sure the next token is predicted
    repositioned_inputs = input_ids[:, :-1]

    results["input_ids"] = repositioned_inputs[0]
    results["attention_mask"] = attention_mask[:, :-1][0]
    results["labels"] = shifted_labels[0]
    return results


class InstructDataset(Dataset):
    def __init__(
        self,
        json_file,
        max_length=2048,
        dataset_size=None,
    ):
        self.indicies: list[tuple[int, int]] = []
        with open(json_file) as fd:
            self.dataset = json.load(fd)["data"]

        for example_idx, example in enumerate(self.dataset):
            for question_idx, _ in enumerate(example["questions"]):
                self.indicies.append((example_idx, question_idx))

        if dataset_size is not None:
            self.indicies = self.indicies[:dataset_size]
        self.max_length = max_length

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, index: int):
        example_idx, question_idx = self.indicies[index]
        example = self.dataset[example_idx]
        processed_examples = preprocess(example, question_idx, self.max_length)
        return processed_examples


if __name__ == "__main__":
    dataset = InstructDataset(
        json_file="/home/chief/instruction-tuning/instruct/dataset/coqa-train-v1.0.json",
    )
    print(dataset[3])
