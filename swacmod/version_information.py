from . import __version__
from . import __commit_id__

def print_version(print=print):
	formatted_version_information = format_version_information()
	print(formatted_version_information)

def format_version_information():
	return f"Surface Water Accounting Model - SWAc\nVersion: {__version__}\nCommit ID: {__commit_id__}\n"
