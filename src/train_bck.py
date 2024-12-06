import os
from pathlib import Path
import logging
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List
from loguru import logger

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper, instantiate_callbacks, instantiate_loggers

log = logging.getLogger(__name__)

def upload_to_s3(file_path: str, bucket_name: str, s3_key: str):
    """
    Upload a file to an S3 bucket.

    Args:
        file_path (str): Local file path to upload.
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): S3 object key (path in the bucket).

    Returns:
        str: URL of the uploaded file.
    """
    s3_client = boto3.client("s3")

    try:
        log.info(f"Uploading {file_path} to S3 bucket {bucket_name}...")
        s3_client.upload_file(file_path, bucket_name, s3_key)
        s3_url = f"s3://{bucket_name}/{s3_key}"
        log.info(f"File uploaded successfully to {s3_url}")
        return s3_url
    except (NoCredentialsError, ClientError) as e:
        log.error(f"Failed to upload file to S3: {e}")
        raise


@task_wrapper
def train(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting training!")
    trainer.fit(model, datamodule)
    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")

    # Save the best model checkpoint to S3
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        log.info(f"Best model found at {best_model_path}")
        s3_bucket = cfg.s3.bucket_name
        s3_key = os.path.join(cfg.s3.key_prefix, os.path.basename(best_model_path))
        upload_to_s3(best_model_path, s3_bucket, s3_key)
    else:
        log.warning("No best model checkpoint found!")


@task_wrapper
def test(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting testing!")
    if trainer.checkpoint_callback.best_model_path:
        log.info(
            f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}"
        )
        test_metrics = trainer.test(
            model, datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path
        )
    else:
        log.warning("No checkpoint found! Using current model weights.")
        test_metrics = trainer.test(model, datamodule)
    log.info(f"Test metrics:\n{test_metrics}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)

    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    if cfg.get("train"):
        train(cfg, trainer, model, datamodule)

    # Test the model
    if cfg.get("test"):
        test(cfg, trainer, model, datamodule)


if __name__ == "__main__":
    main()
