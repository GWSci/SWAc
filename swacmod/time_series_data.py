import ast
import csv

from . import utils as u

class CsvTimeSeriesData:
	def __init__(self, csv_filename):
		print(f"Using CsvTimeSeriesData for file: {csv_filename}")
		try:
			reader = csv.reader(open(csv_filename, "r"))
		except IOError as err:
			message = f"Could not read file: {csv_filename}"
			raise u.InputOutputError(message)
		try:
			rows = [[ast.literal_eval(j) for j in row]
					for row in reader]
			
			print(f"type(rows) = {type(rows)}")
			print(f"type(rows[0]) = {type(rows[0])}")
			print(f"type(rows[0][0]) = {type(rows[0][0])}")

			self.rows = rows
		except IndexError as err:
			message = f"Could not read file: {csv_filename}"
			raise u.InputOutputError(message)

	def row(self, index):
		return self.rows[index]

	def __len__(self):
		return len(self.rows)
