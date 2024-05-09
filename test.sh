discovery_root="test"

source env/bin/activate
python3.11 -c "import swacmod.compile_model"
env TQDM_DISABLE=true coverage run -m unittest discover -s $discovery_root
exit_status=$?
coverage report -m
deactivate
exit $exit_status
