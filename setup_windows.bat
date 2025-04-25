@echo off
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Cannot find Python. Please install it from www.python.org and rerun this script.
    exit /b
)

where pandoc >nul 2>nul
if %errorlevel% neq 0 (
    echo Cannot find Pandoc. Please install it from https://pandoc.org and rerun this script.
    exit /b
)

@echo on
py -3.12 -m venv env || exit /b

call env/Scripts/activate.bat || exit /b
python -m pip install --upgrade pip || exit /b
pip install -r requirements.txt || exit /b
call env/Scripts/deactivate.bat || exit /b

py -3.12 -m venv env-lint || exit /b

call env-lint/Scripts/activate.bat || exit /b
python -m pip install --upgrade pip || exit /b
pip install -r requirements-lint.txt || exit /b
call env-lint/Scripts/deactivate.bat || exit /b
