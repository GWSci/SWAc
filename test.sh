source env/bin/activate
python3.11 -c "import swacmod.compile_model"
coverage run -m unittest discover -s test
exit_status=$?
coverage report -m
deactivate
exit $exit_status
