def create_github_release():
    print("Hello, World!")

def convert_version_to_tag_name(version):
    return "v0.0." + str(version[2])

if (__name__ == "__main__"):
    create_github_release()
