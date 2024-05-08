#! /bin/bash

set -e

brew update
brew install python@3.11

python3.11 -m venv env

source env/bin/activate
pip install -r requirements.txt
deactivate
