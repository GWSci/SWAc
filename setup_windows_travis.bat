choco install python313 -y
choco install pandoc -y

py -3.13 -m venv env || exit /b

call env/Scripts/activate.bat || exit /b
python -m pip install --upgrade pip || exit /b
pip install -r requirements.txt || exit /b
call env/Scripts/deactivate.bat || exit /b
