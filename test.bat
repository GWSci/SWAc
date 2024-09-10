call env/Scripts/activate.bat
python "compile_model.py"
setlocal
set "TQDM_DISABLE=true" && python -m unittest discover -s test
endlocal
call env\Scripts\deactivate.bat
