def create_github_release():
    print("Hello, World!")

def convert_version_to_tag_name(version):
    major = str(version[0])
    minor = str(version[1])
    patch = str(version[2])
    return f"v{major}.{minor}.{patch}"

if (__name__ == "__main__"):
    create_github_release()
