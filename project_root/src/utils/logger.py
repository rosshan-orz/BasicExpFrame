import logging
import os
from pathlib import Path
from typing import Dict, Any, Union
from torch.utils.tensorboard import SummaryWriter
from box import Box

class BaseLogger:
    """
    Encapsulates Python's logging and TensorBoard's SummaryWriter.
    Handles creation of log directories and manages logging output.
    """
    def __init__(self, log_dir: Union[str, Path], config: Box):
        """
        Initialize logger.
        :param log_dir: directory of log
        :param config: config from configuration YAML
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup standard Python logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if logger was already configured
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)

            # File handler
            file_handler = logging.FileHandler(self.log_dir / "train.log")
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        self.info(f"Logging initialized. Log directory: {self.log_dir}")

        # Setup TensorBoard SummaryWriter
        self.writer = SummaryWriter(str(self.log_dir / "tb_logs"))
        self.info(f"TensorBoard SummaryWriter initialized. Events will be saved to: {self.log_dir / 'tb_logs'}")

    def info(self, msg: str) -> None:
        """
        Logs an informational message to both console and the log file.
        :param msg: Message to be logged
        """
        self.logger.info(msg)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Logs a dictionary of scalar metrics to TensorBoard.
        :param metrics: Dictionary of metric names and their corresponding values.
        :param step: step when logging
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            else:
                self.logger.warning(
                    f"Metric '{key}' has non-scalar value '{value}' (type: {type(value)}). Skipping TensorBoard logging for this metric.")
        self.info(f"Logged metrics at step {step}: {metrics}")

    def close(self) -> None:
        """
        Close TensorBoard SummaryWriter and remove file handlers.
        """
        if self.writer:
            self.writer.close()
            self.info("TensorBoard SummaryWriter closed.")

        # Close and remove file handlers to prevent resource leaks
        for handler in list(self.logger.handlers):  # Iterate over a copy to modify in place
            handler.close()
            self.logger.removeHandler(handler)
        self.info("Logger file handlers closed and removed.")

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """

        :param msg:
        :param args:
        :param kwargs:
        :return:
        """
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """

        :param msg:
        :param args:
        :param kwargs:
        :return:
        """
        self.logger.error(msg, *args, **kwargs)