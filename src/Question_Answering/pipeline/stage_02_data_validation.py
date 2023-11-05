from Question_Answering.config.configuration import ConfigurationManager
from Question_Answering.components.data_validation import DataValidation
from Question_Answering.logging import logger


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_files_exist()


if __name__ == "__main":
    STAGE_NAME = "Data Validation stage"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_validation_obj = DataValidationTrainingPipeline()
        data_validation_obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
