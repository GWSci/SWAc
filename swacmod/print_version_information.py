import version_information

def print_version(print=print):
	formatted_version_information = format_version_information()
	print(formatted_version_information)

def format_version_information():
	version = version_information.version
	return f"Surface Water Accounting Model - SWAc\nVersion: {version}\n"
