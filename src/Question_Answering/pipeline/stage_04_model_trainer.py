from Question_Answering.config.configuration import ConfigurationManager
from Question_Answering.components.model_trainer import ModelTrainer
from Question_Answering.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()
