import os
import _version
from github_release_helper import *
import requests
import pprint
import glob

def create_github_release():
    raw_inputs = gather_create_release_raw_inputs()
    release = convert_raw_inputs_to_create_release(raw_inputs)
    print(release)
    release = call_create_release(release)
    return release

def gather_create_release_raw_inputs():
    repo_slug = os.environ.get("TRAVIS_REPO_SLUG", "/")
    version = _version.version
    commit_id = os.environ.get("TRAVIS_COMMIT", "")
    return Create_Release_Raw_Inputs(
        repo_slug = repo_slug,
        version = version,
        commit_id = commit_id,
    )

def call_create_release(params):
    headers = {
        "Authorization": f"Bearer {os.environ.get("RELEASES_TOKEN", "")}",
        "accept": params.accept,
    }
    json = {
        "tag_name": params.tag_name,
        "target_commitish": params.target_commitish,
    }
    data = None
    print(f"{params.url=}")
    print(f"{json=}")
    r = requests.post(params.url, headers = headers, json = json, data = data)
    print(f"{r.status_code=}")
    response_object = r.json()
    pprint.pprint(response_object)
    if (r.status_code == 422 and response_object.get("code", None) == "already_exists"):
        print("Release already exists. Proceeding on the assumption that another parallel process already created it.")
    elif (r.status_code != 201):
        raise Exception("API call to create release failed.")
    return response_object

def call_upload_github_release_asset(release):
    raw_inputs = gather_upload_asset_raw_inputs(release)
    upload = convert_raw_inputs_to_upload_release(raw_inputs)
    call_upload_asset(upload)

def gather_upload_asset_raw_inputs(release):
    glob_match = glob.glob("release/*.zip")
    print(f"{glob_match=}")
    file_path = glob_match[0]
    repo_slug = os.environ.get("TRAVIS_REPO_SLUG", "/")
    release_id = release.get("id", None)
    return Upload_Asset_Raw_Inputs(
        repo_slug = repo_slug,
        release_id = release_id,
        file_path = file_path,
    )

def call_upload_asset(params):
    url = f"https://uploads.github.com/repos/{params.owner}/{params.repo}/releases/{params.release_id}/assets?name={params.name}"
    headers = {
        "Authorization": f"Bearer {os.environ.get("RELEASES_TOKEN", "")}",
        "accept": params.accept,
        "Content-Type": params.content_type,
    }
    file_path = params.file_path
    json = None
    data = data=open(file_path, 'rb')
    print(f"{url=}")
    print(f"{file_path=}")
    r = requests.post(url, headers = headers, json = json, data = data)
    print(f"{r.status_code=}")
    response_object = r.json()
    pprint.pprint(response_object)
    if (r.status_code != 201):
        raise Exception("API call to upload asset failed.")
    return response_object

if (__name__ == "__main__"):
    release = create_github_release()
    call_upload_github_release_asset(release)
