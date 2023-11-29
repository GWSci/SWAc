from contextlib import contextmanager
import csv

@contextmanager
def reader_for(path):
	try:
		with open(path, "r") as csv_file:
			reader = csv.reader(csv_file)
			yield reader
	finally:
		pass
