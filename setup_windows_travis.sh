#! /bin/bash

set -e

choco install python313 -y
choco install pandoc -y

py -m venv env

ls env/Scripts/

env/Scripts/pip install -r requirements.txt
env/Scripts/pip list --outdated
