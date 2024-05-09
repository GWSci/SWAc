import logging

class Environment:
	def print(self, message):
		print(message)

	def set_up_logging(self, path, log_format, level):
		logging.basicConfig(filename=path, format=log_format, level=level)
