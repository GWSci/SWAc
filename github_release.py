import os
import _version

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

    # rganise
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

def parse_repo_slug(slug):
    tokens = slug.split("/")
    owner = tokens[0]
    repo = tokens[1]
    return owner, repo

def convert_version_to_tag_name(version):
    major = str(version[0])
    minor = str(version[1])
    patch = str(version[2])
    return f"v{major}.{minor}.{patch}"

if (__name__ == "__main__"):
    create_github_release()
