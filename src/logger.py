import logging
import os
from datetime import datetime


# Sets up a logger that writes to a file with timestamps.
def setup_logger(name: str, log_dir: str = "logs"):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger




