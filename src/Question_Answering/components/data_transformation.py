import os
from transformers import AutoTokenizer
from datasets import load_from_disk
from Question_Answering.entity import DataTransformationConfig
from Question_Answering.constants import *
from Question_Answering.logging import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.max_length = MAX_LENGTH
        self.stride = STRIDE

    def preprocess_training_data(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions

        return {
            'input_ids': inputs['input_ids'],
            'token_type_ids': inputs['token_type_ids'],
            'attention_mask': inputs['attention_mask'],
            'start_positions': inputs["start_positions"],
            'end_positions': inputs["end_positions"]
        }

    def preprocess_validation_data(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return {
            'input_ids': inputs['input_ids'],
            'token_type_ids': inputs['token_type_ids'],
            'attention_mask': inputs['attention_mask'],
            'offset_mapping': inputs["offset_mapping"],
            'example_id': inputs["example_id"]
        }

    def processing(self):
        train_dataset = load_from_disk(self.config.train_data_path)
        preprocessed_train_dataset = train_dataset.map(self.preprocess_training_data, batched=True,
                                                       remove_columns=train_dataset.column_names)
        preprocessed_train_dataset.save_to_disk(os.path.join(self.config.root_dir, "train_dataset"))
        valid_dataset = load_from_disk(self.config.valid_data_path)
        preprocessed_valid_dataset = valid_dataset.map(self.preprocess_validation_data, batched=True,
                                                       remove_columns=valid_dataset.column_names)
        preprocessed_valid_dataset.save_to_disk(os.path.join(self.config.root_dir, "validation_dataset"))
