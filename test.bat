call env/Scripts/activate.bat
python "compile_model.py"
python -m unittest discover -s test
call env\Scripts\deactivate.bat
