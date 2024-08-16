call env/Scripts/activate.bat
python -c "import compile_model"
python -m unittest discover -s test
call env\Scripts\deactivate.bat
