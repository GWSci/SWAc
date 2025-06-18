#! /bin/bash

set -e

#brew update
#brew install python
#brew install pandoc

/opt/homebrew/opt/python@3.13/libexec/bin/python -m venv env

source env/bin/activate
pip install -r requirements.txt
pip list --outdated
deactivate

/opt/homebrew/opt/python@3.13/libexec/bin/python -m venv env-lint

source env-lint/bin/activate
pip install -r requirements.txt
pip install -r requirements-lint.txt
pip list --outdated
deactivate
