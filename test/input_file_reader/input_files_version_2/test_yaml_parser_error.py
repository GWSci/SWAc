import unittest
from swacmod.input_files.input_files_version_2.loader import load_yaml
from test.input_file_reader.input_files_version_2.mock_file_resource import MockFileResource

class Test_YAML_Parser_Error(unittest.TestCase):
    def test_x(self):
        filename = 'recipe.csv'
        file_contents = 'ingredients:\n   sausage: 1\n  potato: 2\n' # missing one space before potato
        mock_file_opener = MockFileResource.make_mock_file_opener({filename: file_contents})
        load_yaml(filename, mock_file_opener)