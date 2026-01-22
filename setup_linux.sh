#! /bin/bash

set -e

sudo apt-get update
sudo apt-get -y install pandoc

python3 -m venv env

source env/bin/activate
pip install -r requirements.txt
pip list --outdated
deactivate
