import datetime

def timer_log(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)
