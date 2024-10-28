from loss import MrlCrossEntropyLoss
from trainer import DseTrainer as Trainer
from dataset import TrainDataset
from collator import TrainCollator
from arguments import ModelArguments, DataArguments
from dse import DseQwen2
import argparse
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    TrainingArguments,
)


min_pixels = 1 * 28 * 28
max_pixels = 960 * 28 * 28

def get_trainer(json_file: str = None):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=json_file)

    processor = AutoProcessor.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True
    )

    processor.tokenizer.padding_side = "left"

    train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args, processor)
    model = DseQwen2(model_args)  

    loss = MrlCrossEntropyLoss(
        matryoshka_dims=[1536, 1024, 768, 512, 384, 256],
        temperature=model_args.temperature
    )

    trainer = Trainer(
        loss_func=loss,
        model_output_dir=model_args.model_output_dir,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    return (trainer, processor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--local-rank", type=int, required=True)
    args = parser.parse_args()

    trainer, processor = get_trainer(json_file=args.json_file)
    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model()
    