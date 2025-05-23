import sys
import shutil
import os.path
import subprocess
from _version import version
import ast
import git
import os
import argparse
import datetime

version_filename = '_version.py'
commit_id_filename = '_commit_id.py'
build_time_filename = '_build_time.py'

def _default_file_open(filename, how='r'):
        return open(filename, how)

def get_old_version(filename=version_filename, file_open=_default_file_open):
    with file_open(filename, 'r') as file:
        old_version = file.readlines()[0]
    old_version = old_version.replace('\n', '')
    old_version = old_version.replace(' ', '')
    old_version = old_version.replace('version', '')
    old_version = old_version.replace('=', '')
    old_version = ast.literal_eval(old_version)
    return old_version

def get_new_version(old_version):
    new_version = old_version.copy()
    new_version[-1] += 1
    return new_version

def write_new_version(new_version, filename=version_filename, file_open=_default_file_open):
    with file_open(filename, 'w') as file:
        file.write(f'version = {new_version}')

def write_new_commit_id(sha, filename=commit_id_filename, file_open=_default_file_open):
    with file_open(filename, 'w') as file:
        file.write(f'commit_id = "{sha}"')

def format_daytime(date):
    return date.strftime("%d %b %Y %H:%M:%S")

def write_new_build_time(formated_datetime, filename=build_time_filename, file_open=_default_file_open):
    with file_open(filename, 'w') as file:
        file.write(f'build_time = "{formated_datetime}"')

def build():
    if (os.path.exists("build/")):
        shutil.rmtree("build/")
    if (os.path.exists("dist/")):
        shutil.rmtree("dist/")

    if sys.platform == "win32":
        python_binary = "env/Scripts/python"
        zip_input_files_command = "powershell Compress-Archive input_files/*.* dist/input_files.zip"
    else:
        python_binary = "env/bin/python3"
        zip_input_files_command = "zip --quiet --recurse-paths dist/input_files.zip input_files/"

    subprocess.run([python_binary, "compile_model.py"])

    subprocess.run([
        python_binary,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--add-data",
        "./swacmod/specs.yml:./swacmod/",
        "--hidden-import",
        "swacmod.snow_melt",
        "--hidden-import",
        "swacmod.networkx_adaptor",
        "--onefile",
        "swacmod_run.py",
    ])

    subprocess.run(["pandoc", "doc/getting-started.md", "-o", "dist/getting-started.html"])

    subprocess.run(zip_input_files_command, shell=True)

def parse_arguments():
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-f', '--final', action='store_true', 
                        help='The build script will update the version number and push the change to GitHub?')

    return PARSER.parse_args()

def main(args):
    if args.final:
        old_version = get_old_version()
        new_version = get_new_version(old_version)
        write_new_version(new_version)

        repo = git.Repo(os.getcwd())
        repo.git.add(version_filename)
        repo.git.commit('-m', 'updated version number')

        sha = repo.head.object.hexsha
        write_new_commit_id(sha)

        date = datetime.datetime.now()
        formated_datetime = format_daytime(date)
        write_new_build_time(formated_datetime)

        try:
            build()
            repo.git.restore(commit_id_filename)
            repo.git.restore(build_time_filename)
            
            repo.git.push()

        except Exception as err:
            repo.git.reset('HEAD~')
            repo.git.restore(version_filename)
            repo.git.restore(commit_id_filename)
            repo.git.restore(build_time_filename)
            raise Exception(err)
    else:
        build()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
