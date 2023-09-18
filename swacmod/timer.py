import datetime
import time

def make_time_switcher():
	return {"message_to_seconds": {}}

def switch_to(time_switcher, message, time=time):
	time_switcher = switch_off(time_switcher, time=time)
	time_switcher["current_timer"] = start_timing(message, time=time)
	return time_switcher

def switch_off(time_switcher, time=time):
	if "current_timer" in time_switcher:
		timer_just_finished = stop_timing(time_switcher["current_timer"], time=time)
		previous_message = timer_just_finished["message"]
		previous_time = timer_just_finished["elapsed_seconds"]
		if previous_message in time_switcher["message_to_seconds"]:
			seconds_already_logged = time_switcher["message_to_seconds"][previous_message]
		else:
			seconds_already_logged = 0
		time_switcher["message_to_seconds"][previous_message] = seconds_already_logged + previous_time
		del time_switcher["current_timer"]

	return time_switcher

def _time_switcher_report(time_switcher):
	result = []
	for [message, elapsed_seconds] in time_switcher["message_to_seconds"].items():
		row = {
			"message": message,
			"elapsed_seconds": elapsed_seconds,
		}
		result.append(row)
	return result

def print_time_switcher_report(time_switcher):
	tokens = _time_switcher_report(time_switcher)
	print_time_table(tokens)

# Only used in tests
def make_accumulation_timer(message):
	result = {
		"message": message,
		"elapsed_seconds": 0,
	}
	return result

def start_timing(message, time=time):
	seconds_start = time.time()
	return {
		"seconds_start": seconds_start,
		"message" : message,
		"elapsed_seconds": 0,
	}

# Only used in tests
def continue_timing(token, time=time):
	if "seconds_start" in token:
		return token

	seconds_start = time.time()
	token["seconds_start"] = seconds_start
	return token

def stop_timing(token, time=time):
	if not "seconds_start" in token:
		return token

	initial_elapsed_seconds = token["elapsed_seconds"]
	seconds_stop = time.time()
	seconds_start = token["seconds_start"]
	elapsed_seconds = initial_elapsed_seconds + seconds_stop - seconds_start
	token["elapsed_seconds"] = elapsed_seconds
	del token["seconds_start"]
	return token

def log(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)

def print_time_table(tokens):
	rows = make_time_table(tokens)
	for row in rows:
		log(row)

def make_time_table(tokens):
	max_message_length = _find_max_message_length(tokens)
	max_decimal_point_location = _find_max_decimal_point_location(tokens)
	result = []
	for token in tokens:
		line = _format_table_row(token, max_message_length, max_decimal_point_location)
		result.append(line)
	return result

def _find_max_message_length(tokens):
	result = 0
	for token in tokens:
		message_length = len(token["message"])
		if message_length > result:
			result = message_length
	return result

def _find_max_decimal_point_location(tokens):
	result = 0
	for token in tokens:
		seconds = token["elapsed_seconds"]
		seconds_str = str(seconds)
		decimal_point_location = seconds_str.find(".")
		if decimal_point_location > result:
			result = decimal_point_location
	return result

def _format_table_row(token, max_message_length, max_decimal_point_location):
	message = token["message"]
	message_padding = " " * (max_message_length - len(message))
	elapsed_seconds = str(token['elapsed_seconds'])
	time_padding = " " * (max_decimal_point_location - elapsed_seconds.find("."))
	result = f"{message}{message_padding}: {time_padding}{elapsed_seconds}"
	return result
