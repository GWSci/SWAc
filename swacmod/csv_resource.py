from contextlib import contextmanager
import csv

@contextmanager
def reader_for(path):
	try:
		with open(path, "r", encoding = "utf-8-sig") as csv_file:
			reader = csv.reader(csv_file)
			yield reader
	finally:
		pass
