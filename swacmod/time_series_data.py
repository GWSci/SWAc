import ast
import csv
import logging
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
    
try:
    basestring
except NameError:
    basestring = str

from . import utils as u

class TimeSeriesData:
	pass

class CsvTimeSeriesData(TimeSeriesData):
	def __init__(self, param_name, csv_filename):
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

	def __getitem__(self, subscript):
		return self.rows.__getitem__(subscript)

class YamlTimeSeriesData(TimeSeriesData):
	def __init__(self, param_name, csv_filename):
		try:
			rows = load_yaml(csv_filename)[param_name]
			self.rows = rows
		except (IOError, KeyError) as err:
			message = f"Could not read file: {csv_filename}"
			raise u.InputOutputError(message)

	def row(self, index):
		return self.rows[index]

	def __len__(self):
		return len(self.rows)
	
	def __getitem__(self, subscript):
		return self.rows.__getitem__(subscript)


# Copied from input_output.py
def load_yaml(filein):
    """Load a YAML file, lowercase its keys."""
    logging.debug("\t\tLoading %s", filein)

    with open(filein, "r") as fp:
        yml = yaml.load(fp, Loader=Loader)
    try:
        keys = yml.keys()
    except AttributeError:
        return yml

    for key in keys:
        if isinstance(key, basestring):
            if not key.islower():
                new_key = key.lower()
                value = yml.pop(key)
                yml[new_key] = value
    return yml
