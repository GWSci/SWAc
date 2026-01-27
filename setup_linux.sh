#! /bin/bash

set -e

sudo apt-get update

sudo apt-get -y install python3.13
sudo apt-get -y install pandoc

python3.13 -m venv env

ls env/Scripts/

source env/bin/activate
pip install -r requirements.txt
pip list --outdated
deactivate
