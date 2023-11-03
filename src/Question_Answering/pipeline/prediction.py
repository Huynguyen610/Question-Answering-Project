from Question_Answering.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, question, context):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        question_answerer = pipeline("question-answering", model=self.config.model_path, tokenizer=tokenizer)

        print("Context:")
        print(context)
        print("Question:")
        print(question)

        output = question_answerer(question=question, context=context)[0]["answers"]
        print("\nModel Answer:")
        print(output)

        return output