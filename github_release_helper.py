from dataclasses import dataclass
import os

@dataclass
class Create_Release_Raw_Inputs:
    repo_slug: str
    version: str
    commit_id: str

@dataclass
class Create_Release:
    accept: str
    owner: str
    repo: str
    tag_name: str
    target_commitish: str

@dataclass
class Upload_Asset_Raw_Inputs:
    repo_slug: str
    release_id: str
    file_path: str

@dataclass
class Upload_Asset:
    content_type: str
    accept: str
    owner: str
    repo: str
    release_id: str
    name: str
    file_path: str

def convert_raw_inputs_to_create_release(raw_inputs):
    owner, repo = parse_repo_slug(raw_inputs.repo_slug)
    tag_name = convert_version_to_tag_name(raw_inputs.version)

    return Create_Release(
        accept = "application/vnd.github+json",
        owner = owner,
        repo = repo,
        tag_name = tag_name,
        target_commitish = raw_inputs.commit_id,
    )

def convert_raw_inputs_to_upload_release(raw_inputs):
    owner, repo = parse_repo_slug(raw_inputs.repo_slug)
    name = os.path.basename(raw_inputs.file_path)
    return Upload_Asset(
        content_type = "application/zip",
        accept = "application/vnd.github+json",
        owner = owner,
        repo = repo,
        release_id = raw_inputs.release_id,
        name = name,
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
