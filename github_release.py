import os
import _version
from github_release_helper import parse_repo_slug, convert_version_to_tag_name

def create_github_release():
    print("Hello, World!")

def make_create_release():
    # gather
    repo_slug = os.environ["TRAVIS_REPO_SLUG"]
    version = _version.version
    commit_id = os.environ["TRAVIS_COMMIT"]

    # parse
    owner, repo = parse_repo_slug(repo_slug)
    tag_name = convert_version_to_tag_name(version)

    # organise
    header = {
        "accept": "application/vnd.github+json",
    }
    path = {
        "owner": owner,
        "repo": repo,
    }
    body = {
        "tag_name": tag_name,
        "target_commitish": commit_id
    }
    return {
        "header": header,
        "path": path,
        "body": body
    }

if (__name__ == "__main__"):
    create_github_release()
