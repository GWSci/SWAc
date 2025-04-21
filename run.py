import subprocess
import sys

if sys.platform == "win32":
    python_binary = "env/Scripts/python"
else:
    python_binary = "env/bin/python3"

args = sys.argv[1:]

subprocess.run([python_binary, "compile_model.py"])
subprocess.run([python_binary, "swacmod_run.py"] + args)
