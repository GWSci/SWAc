import shutil
import os.path

if (os.path.exists("build/")):
    shutil.rmtree("build/")
if (os.path.exists("dist/")):
    shutil.rmtree("dist/")
