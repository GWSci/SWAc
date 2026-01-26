import unittest
from github_release import convert_version_to_tag_name, parse_repo_slug

class Test_Github_Release(unittest.TestCase):
    def test_convert_version_to_tag_name(self):
        self.assertEqual("v0.0.0", convert_version_to_tag_name([0, 0, 0]))
        self.assertEqual("v0.0.1", convert_version_to_tag_name([0, 0, 1]))
        self.assertEqual("v0.1.0", convert_version_to_tag_name([0, 1, 0]))
        self.assertEqual("v1.0.0", convert_version_to_tag_name([1, 0, 0]))
        self.assertEqual("v2.3.5", convert_version_to_tag_name([2, 3, 5]))

    def test_parse_repo_slug(self):
        owner, repo = parse_repo_slug("aardvark/bat")
        self.assertEqual("aardvark", owner)
        self.assertEqual("bat", repo)

        owner, repo = parse_repo_slug("cat/dog")
        self.assertEqual("cat", owner)
