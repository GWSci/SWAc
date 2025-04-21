import shutil
import os.path
import subprocess

if (os.path.exists("build/")):
    shutil.rmtree("build/")
if (os.path.exists("dist/")):
    shutil.rmtree("dist/")

python_binary = "env/bin/python3"
subprocess.run([python_binary, "compile_model.py"])
