import os
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig , OmegaConf
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List
from loguru import logger

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper , instantiate_callbacks , instantiate_loggers

log = logging.getLogger(__name__)


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