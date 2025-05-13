import os
import swacmod.csv_resource as csv_resource

class DefaultFileResource:
    def _default_file_open(filename):
        return open(filename, "r")
    
    def _default_csv_open(filename):
        return csv_resource.reader_for(filename)

    def _default_filename_exists(filename):
        return os.path.exists(filename)