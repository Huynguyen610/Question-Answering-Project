from Question_Answering.constants import *
from Question_Answering.utils.common import read_yaml, create_directories
from Question_Answering.entity import DataIngestionConfig
from Question_Answering.entity import DataValidationConfig
from Question_Answering.entity import DataTransformationConfig
from Question_Answering.entity import ModelTrainerConfig
from Question_Answering.entity import ModelEvaluationConfig

class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES
        )
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            valid_data_path=config.valid_data_path,
            tokenizer_name=config.tokenizer_name
            )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            valid_data_path=config.valid_data_path,
            model_checkpoint=config.model_checkpoint,
            save_strategy=params.save_strategy,
            num_train_epochs=params.num_train_epochs,
            num_update_step_per_epoch=params.num_update_step_per_epoch,
            batch_size=params.batch_size,
            num_warmup_steps=params.num_warmup_steps,
            logging_steps=params.logging_steps
        )
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.EvaluationArguments

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            raw_valid_data_path=config.raw_valid_data_path,
            train_data_path=config.train_data_path,
            valid_data_path=config.valid_data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name,
            model_checkpoint=config.model_checkpoint
        )
        return model_evaluation_config
