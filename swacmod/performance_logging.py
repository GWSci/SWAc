import logging

# This has been extracted as performance logging should be done at debug level normally
# but possibly at a higher level, or even printed to the console when working on performance.
def log_performance(message):
	logging.debug(message)
