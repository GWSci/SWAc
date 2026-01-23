#! /bin/bash

set -e

choco install python313 -y
choco install pandoc -y

py -m venv env

ls env/bin/

source env/bin/activate
pip install -r requirements.txt
pip list --outdated
deactivate
