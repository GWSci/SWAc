#! /bin/bash

set -e

brew update
brew install python@3.13
brew link python@3.13
brew install pandoc

python3.13 -m venv env

source env/bin/activate
pip install -r requirements.txt
pip list --outdated
deactivate

python3.13 -m venv env-lint

source env-lint/bin/activate
pip install -r requirements.txt
pip install -r requirements-lint.txt
pip list --outdated
deactivate

printf "#!/bin/sh\n\npython3.13 increment_version.py\n" > .git/hooks/pre-commit
