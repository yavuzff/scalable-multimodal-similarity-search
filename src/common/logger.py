"""
Set up logger to output to file and console
"""
import logging
import os
import datetime

# set up log file
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
current_path = os.path.dirname(__file__)
LOG_DIRECTORY = f"{current_path}/../../logs"

# create logs directory if it does not exist
if not os.path.exists(LOG_DIRECTORY):
    print(f"Creating logs directory {LOG_DIRECTORY}")
    os.makedirs(LOG_DIRECTORY)

log_file = f"{current_path}/../../logs/{time}.log"
if not os.path.exists(log_file):
    open(log_file, 'w').close()

# set up logger
logging.basicConfig(filename=log_file, level=logging.INFO, format = '%(asctime)s %(levelname)-8s %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())
