#! /bin/bash

set -e

brew update
brew install python3

python3 -m venv env

source env/bin/activate
pip install -r requirements3.txt
deactivate
