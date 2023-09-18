import ast
import csv
import datetime
import logging
import numpy
import os
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

class Numpy_Dumpy_Time_Series_Data(TimeSeriesData):
	def __init__(self, filename):
		# log("Reading Numpydumpy START")
		shape = convert_numpydumpy_filename_to_shape(filename)
		try:
			self.rows = numpy.memmap(
				filename = filename, 
				dtype = float,
				mode = "r",
				shape = shape)
		except IOError as err:
			message = f"Could not read file: {filename}"
			raise u.InputOutputError(message)
		# log("Reading Numpydumpy END")

	def row(self, index):
		return self.rows[index]

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, subscript):
		return self.rows.__getitem__(subscript)

class CsvTimeSeriesData(TimeSeriesData):
	def __init__(self, param_name, csv_filename):
		log("Reading CSV START")
		try:
			reader = csv.reader(open(csv_filename, "r"))
		except IOError as err:
			message = f"Could not read file: {csv_filename}"
			raise u.InputOutputError(message)
		try:
			rows = [[float(j) for j in row]
					for row in reader]
			
			self.rows = rows
		except IndexError as err:
			message = f"Could not read file: {csv_filename}"
			raise u.InputOutputError(message)
		log("Reading CSV END")

	def row(self, index):
		return self.rows[index]

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, subscript):
		return self.rows.__getitem__(subscript)

class CsvTimeSeriesData_File_Backed(TimeSeriesData):
	def __init__(self, base_path, param_name, csv_filename):
		log("Reading CSV START")
		try:
			reader = csv.reader(open(csv_filename, "r"))
		except IOError as err:
			message = f"Could not read file: {csv_filename}"
			raise u.InputOutputError(message)
		try:
			rows = [[float(j) for j in row]
					for row in reader]
			
			self.rows = convert_rows_to_file_backed_array(base_path, rows, csv_filename)
		except IndexError as err:
			message = f"Could not read file: {csv_filename}"
			raise u.InputOutputError(message)
		log("Reading CSV END")

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

class YamlTimeSeriesData_File_Backed(TimeSeriesData):
	def __init__(self, base_path, param_name, csv_filename):
		try:
			rows = load_yaml(csv_filename)[param_name]
		except (IOError, KeyError) as err:
			message = f"Could not read file: {csv_filename}"
			raise u.InputOutputError(message)
		self.rows = convert_rows_to_file_backed_array(base_path, rows, csv_filename)

	def row(self, index):
		return self.rows[index]

	def __len__(self):
		return len(self.rows)
	
	def __getitem__(self, subscript):
		return self.rows.__getitem__(subscript)

def convert_rows_to_file_backed_array(base_path, rows, csv_filename):
	dtype = calculate_dtype_for_python_list(rows)
	shape = calculate_shape_for_python_list(rows)
	result = numpy.memmap(
		filename = calculate_filename_for_backing_file(base_path, csv_filename, shape), 
		dtype = dtype,
		mode = "w+",
		shape = shape)
	result[:] = rows[:]
	return result

def calculate_filename_for_backing_file(base_path, filename, shape):
	basename = os.path.basename(filename)
	result = f"{base_path}{basename}.{shape}.swacmod_array"
	return result

def calculate_dtype_for_python_list(x):
	if (isinstance(x, float)):
		return float
	if (isinstance(x, list)):
		return calculate_dtype_for_python_list(x[0])
	message = f"Cannot determine dtype for {x}.\n Type = {type(x)}"
	raise Exception(message)

def calculate_shape_for_python_list(x):
	scalars = (float)
	if (isinstance(x[0], scalars)):
		return (len(x))
	if (isinstance(x[0][0], scalars)):
		return (len(x), len(x[0]))
	message = f"Cannot determine shape for {x}.\n Type = {type(x)}"
	raise Exception(message)

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

def calculate_is_in_memory(filename):
	file_size = os.stat(filename).st_size
	memory_cutoff = 300000
	return file_size < memory_cutoff

def convert_bytes_to_human_readable_string(file_size):
	b = file_size
	if b < 1000:
		return f"{b} b"
	kb = int(b / 1024)
	if kb < 1000:
		return f"{kb} kb"
	mb = int(kb / 1024)
	if mb < 1000:
		return f"{mb} kb"
	gb = int(mb / 1024)
	return f"{gb} gb"

def report_using_data_file_backend(filename):
	file_size = os.stat(filename).st_size
	file_size_string = convert_bytes_to_human_readable_string(file_size)
	message = f"Using data file backend. ({file_size_string}) {filename}"
	log(message)

def load_time_series_data(base_path, param, filename, ext):
	is_in_memory = calculate_is_in_memory(filename)
	if ext == "numpydumpy":
		return Numpy_Dumpy_Time_Series_Data(filename)
	elif ext == "csv":
		if is_in_memory:
			return CsvTimeSeriesData(param, filename)
		else:
			report_using_data_file_backend(filename)
			return CsvTimeSeriesData_File_Backed(base_path, param, filename)
	elif ext == "yml":
		if is_in_memory:
			return YamlTimeSeriesData(param, filename)
		else:
			report_using_data_file_backend(filename)
			return YamlTimeSeriesData_File_Backed(base_path, param, filename)
	else:
		log(f"Could not load file: {filename}")

def log(message):
	timestamp = datetime.datetime.now()
	line = f"{timestamp} : {message}"
	print(line)

def convert_numpydumpy_filename_to_shape(filename):
    parts = filename.split(".")
    shape_string = parts[-2]
    shape_tuple = ast.literal_eval(shape_string)
    return shape_tuple
