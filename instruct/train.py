import os
from typing import Any
import torch.nn as nn
import torch
import typer
from dataset import InstructDataset
from peft import LoraConfig, get_peft_model

from transformers import (
    DataCollatorForLanguageModeling,
    LlamaConfig,
    BitsAndBytesConfig,
    LlamaTokenizerFast,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)


pretrained_path: str = f"{os.path.dirname(__file__)}/weights/"
dataset_path: str = (
    "/home/chief/instruction-tuning/instruct/dataset/coqa-train-v1.0.json"
)

MODEL_CONFIG: dict[str, type | Any] = {
    "model": AutoModelForCausalLM,
    "tokenizer": LlamaTokenizerFast.from_pretrained(pretrained_path),
    "config": LlamaConfig.from_pretrained(pretrained_path),
    "collator": DataCollatorForLanguageModeling,
}

if MODEL_CONFIG["tokenizer"].pad_token is None:
    MODEL_CONFIG["tokenizer"].pad_token = MODEL_CONFIG["tokenizer"].eos_token

MODEL_CONFIG["tokenizer"].padding_side = "right"


class CastOutputToFloat(nn.Sequential):
    """Create a custom nn.Sequential module that casts the output to float32."""

    def forward(self, x):
        return super().forward(x).to(torch.float32)


def test_generation(model, test_example):
    example = test_example["input_ids"].unsqueeze(0).to(model.device)
    output_tokens = model.generate(input_ids=example, max_length=256, temperature=0.0)
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    expected_output_text = tokenizer.decode(
        test_example["labels"], skip_special_tokens=True
    )
    print("Output", output_text)
    print("Expected Output", expected_output_text)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(
    experiment_name: str,
    data_file: str = dataset_path,
    model_path: str = pretrained_path,
    dataset_size: int|None = None,
    dataloader_num_workers: int = 0,
    max_length: int = 512,
    batch_size: int = 10,
    dropout: float = 0.0,
    gradient_checkpointing: bool = False,
    pretrained_checkpoint: str | None = None,
    num_train_epochs: float = 10.0,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 4,
    resume: bool = False,
):
    # TODO: start training from random initialization
    # TODO: incorporate other objectives

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # halves the size of the mdoel
        bnb_4bit_use_double_quant=True,
    )

    model_cls: AutoModelForCausalLM = MODEL_CONFIG["model"]
    model = model_cls.from_pretrained(model_path, quantization_config=quant_config)

    print(model)
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    model.lm_head = CastOutputToFloat(model.lm_head)

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    args = TrainingArguments(
        output_dir=experiment_name,
        run_name=experiment_name,
        dataloader_num_workers=dataloader_num_workers,
        per_device_train_batch_size=batch_size,
        do_eval=False,
        evaluation_strategy="no",
        num_train_epochs=num_train_epochs,
        prediction_loss_only=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ignore_data_skip=False,
        save_steps=100,
        save_total_limit=2,
        save_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        report_to="wandb",
        # gradient_checkpointing=gradient_checkpointing,
        fp16=True,
    )

    collator = MODEL_CONFIG["collator"](
        tokenizer=MODEL_CONFIG["tokenizer"],
        return_tensors="pt",
        mlm=False,
    )

    train_dataset = InstructDataset(
        json_file=data_file, max_length=max_length, dataset_size=dataset_size
    )
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer = Trainer(**trainer_kwargs)

    if resume:
        trainer.train(resume_from_checkpoint = resume)
    else:
        trainer.train()
    # test_generation(model, test_example=train_dataset[0])
    trainer.save_model()


if __name__ == "__main__":
    typer.run(main)
