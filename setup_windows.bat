@echo off
where python
if %errorlevel% neq 0 (
    echo Cannot find Python. Please install it from www.python.org and rerun this script.
    exit /b
)

@echo on
python -m venv env || exit /b

call env/Scripts/activate.bat
pip install -r requirements3.txt
deactivate