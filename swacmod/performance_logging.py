import datetime

def log_performance(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)
