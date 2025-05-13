import io

class MockFileResource:
    def make_mock_file_opener(filenames_to_contents):
        return lambda filename: io.StringIO(filenames_to_contents[filename])

    def make_mock_filename_exists(directory):
        return lambda filename: filename in directory