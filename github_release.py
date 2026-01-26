import os
import _version
from github_release_helper import *
import requests
import pprint

def create_github_release():
    raw_inputs = gather_create_release_raw_inputs()
    release = convert_raw_inputs_to_create_release(raw_inputs)
    print(release)
    call_create_release(release)

def gather_create_release_raw_inputs():
    repo_slug = os.environ.get("TRAVIS_REPO_SLUG", "/")
    version = _version.version
    commit_id = os.environ.get("TRAVIS_COMMIT", "")
    return Create_Release_Raw_Inputs(
        repo_slug = repo_slug,
        version = version,
        commit_id = commit_id,
    )

def call_create_release(release):
    url = f"https://api.github.com/repos/{release.owner}/{release.repo}/releases"
    headers = {
        "Authorization": f"Bearer {os.environ.get("RELEASES_TOKEN", "")}",
        "accept": release.accept
    }
    body = {
        "tag_name": release.tag_name,
        "target_commitish": release.target_commitish,
    }
    print(f"{url=}")
    print(f"{body=}")
    r = requests.post(url, headers = headers, json = body)
    print(f"{r.status_code=}")
    response_object = r.json()
    pprint.pprint(response_object)
    if (r.status_code != 201):
        raise Exception("API call to create release failed.")
    return response_object

if (__name__ == "__main__"):
    create_github_release()
