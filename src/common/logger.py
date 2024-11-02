"""
Set up logger to output to file and console
"""
import logging
import os
import datetime

# set up log file
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file = f"../../logs/{time}.log"
if not os.path.exists(log_file):
    open(log_file, 'w').close()

# set up logger
logging.basicConfig(filename=log_file, level=logging.INFO, format = '%(asctime)s %(levelname)-8s %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())
