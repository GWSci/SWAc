def slurp(file_path_string):
	with open(file_path_string) as file:
		contents = file.read()
	return contents

def slurp_without_first_line(file_path_string):
	contents = slurp(file_path_string)
	return contents.split("\n", 1)[1]
