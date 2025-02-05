import logging
from logging import handlers
from datetime import datetime
import os

# log file name
log_file_name = f"allinone_application.log"

# log folder path. if folder not exist then create one
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

# log file name with actual path
log_file_path = os.path.join(log_path, log_file_name)

# creating object
logger = logging.getLogger()

if not logger.hasHandlers():  
    formatter = logging.Formatter("[ %(asctime)s ] [ Line Number: %(lineno)d in %(filename)s ] - [%(levelname)s] - %(message)s")
    handler = handlers.TimedRotatingFileHandler(log_file_path, when="M", backupCount=4)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("INFO")
