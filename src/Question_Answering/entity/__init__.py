from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    valid_data_path: Path
    tokenizer_name: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    valid_data_path: Path
    model_checkpoint: Path
    save_strategy: str
    num_train_epochs: int
    num_update_step_per_epoch: int
    batch_size: int
    num_warmup_steps: int
    logging_steps: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    raw_valid_data_path: Path
    train_data_path: Path
    valid_data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path