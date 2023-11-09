import argparse

from Question_Answering.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Question_Answering.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from Question_Answering.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from Question_Answering.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from Question_Answering.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from Question_Answering.logging import logger


def run(stage_name: str):
    if stage_name == "Data Ingestion stage":
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            data_ingestion = DataIngestionTrainingPipeline()
            data_ingestion.main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e

    elif stage_name == "Data Validation stage":
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            data_validation = DataValidationTrainingPipeline()
            data_validation.main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e

    elif stage_name == "Data Transformation stage":
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            data_transformation = DataTransformationTrainingPipeline()
            data_transformation.main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            raise e

    elif stage_name == "Model Trainer stage":
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            model_trainer = ModelTrainerTrainingPipeline()
            model_trainer.main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            raise e

    elif stage_name == "Model Evaluation stage":
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            model_evaluation = ModelEvaluationTrainingPipeline()
            model_evaluation.main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            raise e

    else:
        print("You need to input a stage")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--stage", default=None)
    parsed_args = args.parse_args()
    run(stage_name=parsed_args.stage)
