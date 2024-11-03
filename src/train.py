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
from typing import List, Dict, Any



# @task_wrapper
# def train(
#     cfg: DictConfig,
#     trainer: L.Trainer,
#     model: L.LightningModule,
#     datamodule: L.LightningDataModule,
# ):
#     log.info("Starting training!")
#     trainer.fit(model, datamodule)
#     train_metrics = trainer.callback_metrics
#     log.info(f"Training metrics:\n{train_metrics}")


# @task_wrapper
# def test(
#     cfg: DictConfig,
#     trainer: L.Trainer,
#     model: L.LightningModule,
#     datamodule: L.LightningDataModule,
# ):
#     log.info("Starting testing!")
#     if trainer.checkpoint_callback.best_model_path:
#         log.info(
#             f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}"
#         )
#         test_metrics = trainer.test(
#             model, datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path
#         )
#     else:
#         log.warning("No checkpoint found! Using current model weights.")
#         test_metrics = trainer.test(model, datamodule)
#     log.info(f"Test metrics:\n{test_metrics}")


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


@task_wrapper
def train(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
) -> Dict[str, Any]:
    log.info("Starting training!")
    trainer.fit(model, datamodule)
    train_metrics = trainer.callback_metrics
    log.info(f"Training metrics:\n{train_metrics}")
    return train_metrics


@task_wrapper
def test(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
) -> Dict[str, Any]:
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
    return test_metrics[0] if test_metrics else {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> float:
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)
    print(cfg)

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
    train_metrics = {}
    if cfg.get("train"):
        train_metrics = train(cfg, trainer, model, datamodule)

    # Test the model
    test_metrics = {}
    if cfg.get("test"):
        test_metrics = test(cfg, trainer, model, datamodule)

    # Combine metrics
    all_metrics = {**train_metrics, **test_metrics}

    # Extract and return the optimization metric
    optimization_metric = all_metrics.get(cfg.get("optimization_metric"))
    if optimization_metric is None:
        log.warning(f"Optimization metric '{cfg.get('optimization_metric')}' not found in metrics. Returning 0.")
        return 0.0
    
    return optimization_metric


if __name__ == "__main__":
    main()