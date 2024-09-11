#! /bin/bash

set -e

brew update
brew install python
brew install pandoc

python3 -m venv env

source env/bin/activate
pip install -r requirements.txt
pip list --outdated
deactivate
