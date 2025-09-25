import logging
import os
import datetime
from logging import handlers
from torch.utils.tensorboard import SummaryWriter
import torch

class ExperimentLogger:
    def __init__(self, log_dir, exp_name="exp", log_level=logging.INFO, config=None):
        """
        Logger that writes to console, rotating log file, and TensorBoard.
        
        Args:
            log_dir (str): directory to save logs
            exp_name (str): experiment name
            log_level: logging level
            config (dict): optional experiment config to log
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{exp_name}_{timestamp}"
        self.log_dir = os.path.join(log_dir, self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # ----- Python logging -----
        self.logger = logging.getLogger(self.exp_name)
        if not self.logger.handlers:  # avoid duplicate handlers
            self.logger.setLevel(log_level)

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(log_level)

            # File handler with rotation
            log_file = os.path.join(self.log_dir, f"{self.exp_name}.log")
            fh = handlers.RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
            fh.setLevel(log_level)

            # Formatter
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)

            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        # ----- TensorBoard -----
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)

        # ----- Log config + device info -----
        if config:
            self.log_config(config)

    def log_config(self, config):
        """Log experiment config + device info."""
        self.logger.info("===== Experiment Configuration =====")
        for k, v in config.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("===== Device Info =====")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def log_scalar(self, tag, value, step):
        """Log a scalar value to TensorBoard."""
        self.tb_writer.add_scalar(tag, value, step)

    def close(self):
        """Close TensorBoard writer."""
        self.tb_writer.close()
