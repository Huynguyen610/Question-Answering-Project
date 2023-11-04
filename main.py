from Question_Answering.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Question_Answering.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from Question_Answering.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from Question_Answering.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from Question_Answering.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from Question_Answering.pipeline.prediction import PredictionPipeline
from Question_Answering.logging import logger

if __name__ =="__main__":

    """
STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
"""
"""
STAGE_NAME = "Data Validation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
"""
"""
STAGE_NAME = "Data Transformation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise e
"""
"""
STAGE_NAME = "Model Trainer stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise e
"""
"""
STAGE_NAME = "Model Evaluation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise e
"""

STAGE_NAME = "Prediction stage"
context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50"
question = 'Which NFL team represented the AFC at Super Bowl 50?'

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    Prediction = PredictionPipeline()
    Prediction.predict(question=question, context=context)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise e