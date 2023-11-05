from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_from_disk
import torch
import evaluate
from Question_Answering.entity import ModelEvaluationConfig
from collections import defaultdict
import numpy as np
from accelerate import Accelerator
from transformers import default_data_collator
from torch.utils.data import DataLoader


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def compute_metrics(self, start_logits, end_logits, features, examples):
        # a function to compute metrics after we have finished our predict
        metric = evaluate.load("squad")
        example_to_features = defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        n_best = 20
        max_answer_length = 30
        predicted_answers = []

        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
                end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                                end_index < start_index
                                or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return self.metric.compute(predictions=predicted_answers, references=theoretical_answers)

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)
        model = AutoModelForQuestionAnswering.from_pretrained(self.config.model_path).to(device)

        raw_datasets_validation_split = load_from_disk(self.config.raw_valid_data_path)
        valid_dataset = load_from_disk(self.config.valid_data_path)
        valid_dataset_for_model = valid_dataset.remove_columns(["example_id", "offset_mapping"])
        valid_dataset.set_format("torch")
        accelerator = Accelerator()

        eval_dataloader = DataLoader(
            valid_dataset_for_model,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=8
        )

        model.eval()
        start_logits = []
        end_logits = []

        for batch in tqdm(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(valid_dataset)]
        end_logits = end_logits[: len(valid_dataset)]
        metrics = self.compute_metrics(
            start_logits, end_logits, valid_dataset, raw_datasets_validation_split)
        print(metrics)
        with open(self.config.metric_file_name, 'w') as f:
            f.write(f"{metrics}")
