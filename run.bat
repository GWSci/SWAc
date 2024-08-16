call env/Scripts/activate.bat
python -c "import compile_model"
python swacmod_run.py %*
deactivate
