import shutil
import os.path
import subprocess

if (os.path.exists("build/")):
    shutil.rmtree("build/")
if (os.path.exists("dist/")):
    shutil.rmtree("dist/")

python_binary = "env/bin/python3"

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
