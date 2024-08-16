call env/Scripts/activate.bat
python -c "import swacmod.compile_model"
python swacmod_run.py %*
deactivate
