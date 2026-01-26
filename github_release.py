def create_github_release():
    print("Hello, World!")

def convert_version_to_tag_name(version):
    patch = str(version[2])
    return f"v0.0.{patch}"

if (__name__ == "__main__"):
    create_github_release()
