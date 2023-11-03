from Question_Answering.config.configuration import ConfigurationManager
from Question_Answering.components.model_evaluation import ModelEvaluation
from Question_Answering.logging import logger


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()
