from . import __version__
from . import __commit_id__
from . import __build_time__

def format_version_information():
	return f"Surface Water Accounting Model - SWAc\nVersion: {__version__}\nCommit ID: {__commit_id__}\nCompiled on {__build_time__}\n"
