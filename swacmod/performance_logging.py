import datetime

def timer_log(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)

def time_series_data_log(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)
