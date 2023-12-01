python -m venv env || exit /b

call env/Scripts/activate.bat
pip install -r requirements3.txt
deactivate