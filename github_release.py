import os
import _version
from github_release_helper import *

def create_github_release():
    raw_inputs = gather_create_release_raw_inputs()
    create_release = convert_raw_inputs_to_create_release(raw_inputs)
    print(create_release)

def gather_create_release_raw_inputs():
    repo_slug = os.environ.get("TRAVIS_REPO_SLUG", "/")
    version = _version.version
    commit_id = os.environ.get("TRAVIS_COMMIT", "")
    return Create_Release_Raw_Inputs(
        repo_slug = repo_slug,
        version = version,
        commit_id = commit_id,
    )

if (__name__ == "__main__"):
    create_github_release()
