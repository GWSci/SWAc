import datetime
import logging

def log_performance(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)
	logging.debug(message)
