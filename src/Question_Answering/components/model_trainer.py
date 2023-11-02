from transformers import TrainingArguments, Trainer
from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import os
from Question_Answering.entity import ModelTrainerConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_dataset = load_from_disk(self.config.train_data_path)
        train_dataset.set_format("torch")

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.config.batch_size,
        )

        model = AutoModelForQuestionAnswering.from_pretrained(self.config.model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        accelerator = Accelerator(fp16=True)
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader)

        num_train_epochs = self.config.num_train_epochs
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        progress_bar = tqdm(range(num_training_steps))

        for epoch in range(num_train_epochs):
            # Training
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(self.config.root_dir, "Question-Answering-model",
                                            save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(self.config.root_dir, "tokenizer")