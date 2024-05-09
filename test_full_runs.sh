source env/bin/activate
python3.11 -c "import swacmod.compile_model"
env TQDM_DISABLE=true coverage run -m unittest discover -s .
exit_status=$?
coverage report -m
pip list --outdated
deactivate
exit $exit_status
