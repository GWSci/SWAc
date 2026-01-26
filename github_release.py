import os
import _version

def create_github_release():
    print("Hello, World!")

def make_create_release():
    owner, repo = parse_repo_slug(os.environ["TRAVIS_REPO_SLUG"])
    header = {
        "accept": "application/vnd.github+json",
    }
    path = {
        "owner": owner,
        "repo": repo,
    }
    body = {
        "tag_name": convert_version_to_tag_name(_version.version),
        "target_commitish": os.environ["TRAVIS_COMMIT"]
    }
    return {
        "header": header,
        "path": path,
        "body": body
    }

def parse_repo_slug(slug):
    tokens = slug.split("/")
    return tokens[0], "bat"

def convert_version_to_tag_name(version):
    major = str(version[0])
    minor = str(version[1])
    patch = str(version[2])
    return f"v{major}.{minor}.{patch}"

if (__name__ == "__main__"):
    create_github_release()
