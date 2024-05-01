call env/Scripts/activate.bat
python -c "import swacmod.compile_model"
python -m unittest discover -s .
call env\Scripts\deactivate.bat
