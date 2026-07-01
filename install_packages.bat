@echo off
color 0E
echo.
echo ============================================================
echo   INSTALL PYTHON PACKAGES FOR LOAN APPROVAL PROJECT
echo ============================================================
echo.
echo This script will install:
echo  - pandas, numpy, matplotlib, seaborn
echo  - scikit-learn, imbalanced-learn
echo  - xgboost, fpdf2, joblib, jupyter
echo.

REM ── Detect Python ─────────────────────────────────────────────
set PYTHON_EXE=

REM Check Anaconda locations first (most common on data science machines)
for %%P in (
    "C:\Users\%USERNAME%\anaconda3\python.exe"
    "C:\Users\%USERNAME%\Anaconda3\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
    "C:\anaconda3\python.exe"
    "C:\Users\%USERNAME%\miniconda3\python.exe"
    "C:\ProgramData\miniconda3\python.exe"
    "C:\miniconda3\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "C:\Python39\python.exe"
    "C:\Python38\python.exe"
) do (
    if exist %%P (
        if "%PYTHON_EXE%"=="" set PYTHON_EXE=%%~P
    )
)

REM If still not found, try PATH
if "%PYTHON_EXE%"=="" (
    python --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_EXE=python
    )
)

if "%PYTHON_EXE%"=="" (
    echo.
    echo ============================================================
    echo  ERROR: Python is NOT installed on this computer!
    echo ============================================================
    echo.
    echo Please install Python first, then re-run this script.
    echo.
    echo RECOMMENDED: Install Anaconda (includes everything)
    echo   URL: https://www.anaconda.com/download
    echo   - Click Download, run installer
    echo   - Check "Add Anaconda to PATH" during install
    echo   - After install, RESTART this terminal, then run again
    echo.
    echo OR install Python from python.org:
    echo   URL: https://www.python.org/downloads/
    echo   - Check "Add Python to PATH" during install
    echo.
    pause
    exit /b 1
)

echo Found Python: %PYTHON_EXE%
%PYTHON_EXE% --version
echo.

echo Installing packages (this may take 2-5 minutes)...
echo.

%PYTHON_EXE% -m pip install --upgrade pip
%PYTHON_EXE% -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn fpdf2 joblib jupyter notebook
%PYTHON_EXE% -m pip install xgboost

echo.
echo Verifying...
%PYTHON_EXE% -c "import pandas, numpy, matplotlib, seaborn, sklearn, imblearn, fpdf; print('SUCCESS: All packages installed correctly!')"

echo.
echo ============================================================
echo  Now run the project with:
echo     %PYTHON_EXE% run_project.py
echo.
echo  Or open Jupyter Notebook:
echo     %PYTHON_EXE% -m jupyter notebook loan_approval_prediction.ipynb
echo ============================================================
echo.

set /p RUNOW="Run project now? (y/n): "
if /i "%RUNOW%"=="y" (
    echo.
    %PYTHON_EXE% run_project.py
)

pause
