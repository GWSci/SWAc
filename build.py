import sys
import shutil
import os.path
import subprocess

if (os.path.exists("build/")):
    shutil.rmtree("build/")
if (os.path.exists("dist/")):
    shutil.rmtree("dist/")

if sys.platform == "win32":
    python_binary = "env/Scripts/python"
    zip_input_files_command = "powershell Compress-Archive input_files\*.* dist\input_files.zip"
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
