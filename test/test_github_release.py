import unittest
from github_release_helper import *

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
        self.assertEqual("dog", repo)

    def test_convert_raw_inputs_to_create_release(self):
        raw_inputs = Create_Release_Raw_Inputs(
            repo_slug = "aardvark_owner/bat_repo",
            version = [2, 3, 5],
            commit_id = "cat_commit_id"
        )
        expected = Create_Release(
            accept = "application/vnd.github+json",
            owner = "aardvark_owner",
            repo = "bat_repo",
            tag_name = "v2.3.5",
            target_commitish = "cat_commit_id",
        )
        actual = convert_raw_inputs_to_create_release(raw_inputs)
        self.assertEqual(expected, actual)

    def test_convert_raw_inputs_to_upload_release(self):
        raw_inputs = Upload_Asset_Raw_Inputs(
            repo_slug = "aardvark_owner/bat_repo",
            release_id = "cat_release_id",
            file_path = "dog/elephant/fox.txt",
        )
        expected = Upload_Asset(
            content_type = "application/zip",
            accept = "",
            owner = "",
            repo = "",
            release_id = "",
            name = "",
        )
        actual = convert_raw_inputs_to_upload_release(raw_inputs)
        self.assertEqual(expected, actual)
