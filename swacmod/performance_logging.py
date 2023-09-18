import datetime
import swacmod.feature_flags as ff

def timer_log(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)

def time_series_data_log(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)

def swacmod_run_log(message):
    if ff.use_extra_logging:
        timestamp = datetime.datetime.now()
        line = f"{timestamp} : swacmod_run.py : {message}"
        print(line)
