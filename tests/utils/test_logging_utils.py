import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig
from functools import wraps

from src.utils.logging_utils import *

@pytest.fixture
def mock_logger():
    # Fixture to reset logger before each test
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    yield
    logger.remove()

def test_setup_logger(tmp_path, mock_logger):
    log_file = tmp_path / "test_log.log"
    setup_logger(log_file)

    assert log_file.exists(), "Log file should be created"
    logger.info("Test log")
    with open(log_file, 'r') as f:
        log_content = f.read()
    assert "Test log" in log_content, "Log should be written to file"

def test_task_wrapper_success(mock_logger):
    @task_wrapper
    def test_func(a, b):
        return a + b

    result = test_func(2, 3)
    assert result == 5, "Wrapped function should return correct result"

def test_task_wrapper_exception(mock_logger):
    @task_wrapper
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        test_func()

def test_get_rich_progress():
    progress = get_rich_progress()
    assert isinstance(progress, Progress), "Should return a Progress instance"
    assert len(progress.columns) > 0, "Progress should have columns"

@patch('hydra.utils.instantiate')
def test_instantiate_callbacks(mock_instantiate):
    # Create mock DictConfig
    callback_cfg = DictConfig({
        "cb1": {"_target_": "MockCallback1"},
        "cb2": {"_target_": "MockCallback2"}
    })
    
    mock_callback = MagicMock()
    mock_instantiate.return_value = mock_callback
    
    callbacks = instantiate_callbacks(callback_cfg)
    
    assert len(callbacks) == 2, "Should instantiate two callbacks"
    assert mock_instantiate.call_count == 2, "Instantiate should be called for each callback"

@patch('hydra.utils.instantiate')
def test_instantiate_callbacks_empty(mock_instantiate):
    # Create empty DictConfig
    callback_cfg = DictConfig({})
    
    callbacks = instantiate_callbacks(callback_cfg)
    
    assert len(callbacks) == 0, "No callbacks should be instantiated for empty config"

@patch('hydra.utils.instantiate')
def test_instantiate_loggers(mock_instantiate):
    # Create mock DictConfig
    logger_cfg = DictConfig({
        "lg1": {"_target_": "MockLogger1"},
        "lg2": {"_target_": "MockLogger2"}
    })
    
    mock_logger = MagicMock()
    mock_instantiate.return_value = mock_logger
    
    loggers = instantiate_loggers(logger_cfg)
    
    assert len(loggers) == 2, "Should instantiate two loggers"
    assert mock_instantiate.call_count == 2, "Instantiate should be called for each logger"

@patch('hydra.utils.instantiate')
def test_instantiate_loggers_empty(mock_instantiate):
    # Create empty DictConfig
    logger_cfg = DictConfig({})
    
    loggers = instantiate_loggers(logger_cfg)
    
    assert len(loggers) == 0, "No loggers should be instantiated for empty config"
