import io
import csv

def assert_csv_similar(expected, actual):
	expected_grid = _read_csv(expected)
	actual_grid = _read_csv(actual)
	error_messages = []

	expected_row_count = len(expected_grid)
	actual_row_count = len(actual_grid)
	if (expected_row_count != actual_row_count):
		error_messages.append(f"Difference in row counts. Expected: {expected_row_count} Actual: {actual_row_count}")

	for row_index in range(min(expected_row_count, actual_row_count)):
		
		expected_column_count = len(expected_grid[row_index])
		actual_column_count = len(actual_grid[row_index])
		if (expected_column_count != actual_column_count):
			error_messages.append(f"Difference in column count for row={row_index}. Expected: {expected_column_count} Actual: {actual_column_count}")

		for col_index in range(min(expected_column_count, actual_column_count)):
			expected_cell = expected_grid[row_index][col_index]
			actual_cell = actual_grid[row_index][col_index]
			if (not _are_cells_close(expected_cell, actual_cell)):
				message = f"Difference in row={row_index}, col={col_index}. Expected: {expected_cell} Actual: {actual_cell}"
				error_messages.append(message)

	if (len(error_messages) > 0):
		raise AssertionError(error_messages)

def _are_cells_close(a, b):
	if (_is_float(a) and _is_float(b)):
		return abs(float(b) - float(a)) < 0.00001
	else:
		return a == b

def _is_float(x):
	try:
		float(x)
		return True
	except ValueError:
		return False

def _read_csv(file_contents):
	file = io.StringIO(file_contents)
	csv_reader = csv.reader(file)
	result = []
	for row in csv_reader:
		result.append(row)
	return result
