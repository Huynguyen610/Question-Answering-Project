from Question_Answering.config.configuration import ConfigurationManager
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, question, context):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModelForQuestionAnswering.from_pretrained(self.config.model_path).to(device)
        inputs = tokenizer(question, context, return_tensors="pt")
        print("Context:")
        print(context)
        print("Question:")
        print(question)
        outputs = model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        answers = tokenizer.decode(predict_answer_tokens)
        print("\nModel Answer:")
        print(answers)
        return answers
