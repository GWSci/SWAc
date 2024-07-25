import psutil

def memory_info_for_pid(pid):
	process = psutil.Process(pid)
	return process.memory_info()
