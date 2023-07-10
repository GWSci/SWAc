import datetime
import time

def report_time(message, function):
	token = start_timing(message)
	result = function()
	stop_timing(message, token)
	return result

def start_timing(message):
	seconds_start = time.time()
	log(f"{message} START")
	return {
		"seconds_start": seconds_start,
		"message" : message,
	}

def stop_timing(token):
	message = token["message"]
	log(f"{message} STOP")
	seconds_stop = time.time()
	seconds_start = token["seconds_start"]
	elapsed_seconds = seconds_stop - seconds_start
	log(f"{message}: {elapsed_seconds} s")

def log(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)
