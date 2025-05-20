from . import __version__
from . import __commit_id__
from . import __build_time__

def format_version_information():
	return f"Surface Water Accounting Model - SWAc\n \
			Version: {__version__}\n \
			Commit ID: {__commit_id__}\n \
			Compiled on {__build_time__}\n"
