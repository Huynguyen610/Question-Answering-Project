from Question_Answering.config.configuration import ConfigurationManager
from Question_Answering.components.data_transformation import DataTransformation
from Question_Answering.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.processing()


if __name__ == "__main":
    STAGE_NAME = "Data Transformation stage"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transformation_obj = DataTransformationTrainingPipeline()
        data_transformation_obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise e
