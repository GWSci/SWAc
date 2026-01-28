from dataclasses import dataclass
import os

@dataclass
class Create_Release_Raw_Inputs:
    repo_slug: str
    version: str
    commit_id: str

@dataclass
class Create_Release:
    url: str
    accept: str
    tag_name: str
    target_commitish: str

@dataclass
class Upload_Asset_Raw_Inputs:
    repo_slug: str
    release_id: str
    file_path: str

@dataclass
class Upload_Asset:
    url: str
    content_type: str
    accept: str
    file_path: str

def convert_raw_inputs_to_create_release(raw_inputs):
    owner, repo = parse_repo_slug(raw_inputs.repo_slug)
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    tag_name = convert_version_to_tag_name(raw_inputs.version)

    return Create_Release(
        url = url,
        accept = "application/vnd.github+json",
        tag_name = tag_name,
        target_commitish = raw_inputs.commit_id,
    )

def convert_raw_inputs_to_upload_release(raw_inputs):
    owner, repo = parse_repo_slug(raw_inputs.repo_slug)
    name = os.path.basename(raw_inputs.file_path)
    url = f"https://uploads.github.com/repos/{owner}/{repo}/releases/{raw_inputs.release_id}/assets?name={name}"
    return Upload_Asset(
        url = url,
        content_type = "application/zip",
        accept = "application/vnd.github+json",
        file_path = raw_inputs.file_path,
    )

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
