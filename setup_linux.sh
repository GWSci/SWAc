#! /bin/bash

set -e

python3.11 -m venv env

source env/bin/activate
pip install -r requirements3.txt
deactivate
