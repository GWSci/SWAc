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
