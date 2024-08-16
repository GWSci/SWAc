call env/Scripts/activate.bat
python "compile_model.py"
python swacmod_run.py %*
deactivate
