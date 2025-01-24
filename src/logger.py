import logging
from datetime import datetime
import os

# log file name
log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H:%M:%S')}.log"

# log folder path. if folder not exist then create one
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

# log file name with actual path
log_file_path = os.path.join(log_path, log_file_name)

# creating object
logging.basicConfig(
    filename = log_file_path,
    format = "[ %(asctime)s ] Line Number: %(lineno)d in %(name)s - %(levelname)s - %(message)s",
    level = "INFO"
)
