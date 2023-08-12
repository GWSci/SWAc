import datetime
import numpy
import time

def report_time(message, function):
	token = start_timing(message)
	result = function()
	stop_timing(token)
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
	token["seconds_stop"] = seconds_stop
	token["elapsed_seconds"] = elapsed_seconds
	log(f"{message}: {elapsed_seconds} s")
	return token

def log(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)

def report_array(name, arr):
	gb = arr.nbytes / 1024 / 1024 / 1024
	message = f"{name} {arr.dtype} {arr.shape} {gb} gb"
	log(message)

def report_frequencies(message, arr):
	values, counts = numpy.unique(arr, return_counts=True)
	counts = numpy.sort(counts)
	counts = numpy.flip(counts)
	unique_value_count = len(values)
	array_length = len(arr)
	log(f"Frequencies: {message} : {unique_value_count} unique values out of {array_length}.")
	log(f"Frequencies: {message} : Counts: {counts.tolist()}")

def make_time_table(tokens):
	max_message_length = _find_max_message_length(tokens)
	result = []
	for token in tokens:
		message = token["message"]
		while len(message) < max_message_length:
			message += " "
		line = f"{message}: {token['elapsed_seconds']}"
		result.append(line)
	return result

def _find_max_message_length(tokens):
	result = 0
	for token in tokens:
		message_length = len(token["message"])
		if message_length > result:
			result = message_length
	return result
