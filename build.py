import sys
import shutil
import os.path
import subprocess
from _version import version
import ast

args = sys.argv[1:]

def _default_file_open(filename):
        return open(filename, "r")

def _default_file_write(filename):
        return open(filename, "w")

def update_version(filename = '_version.py', file_open = _default_file_open, file_write=_default_file_write):
    with file_open(filename) as file:
        old_version = file.readlines()[0]
    old_version = old_version.replace('\n', '')
    old_version = old_version.replace(' ', '')
    old_version = old_version.replace('version', '')
    old_version = old_version.replace('=', '')
    old_version = ast.literal_eval(old_version)
    new_version = old_version.copy()
    new_version[-1] += 1

def main(args):
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

if __name__ == '__main__':
    main(args)
