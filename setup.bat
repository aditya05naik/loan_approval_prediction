@echo off
color 0A
echo.
echo ============================================================
echo   LOAN APPROVAL PREDICTION - ENVIRONMENT SETUP
echo ============================================================
echo.

REM ── Step 1: Detect Python ─────────────────────────────────────
set PYTHON_EXE=
set PIP_EXE=

REM Check common install locations
if exist "C:\Users\%USERNAME%\anaconda3\python.exe"   set PYTHON_EXE=C:\Users\%USERNAME%\anaconda3\python.exe
if exist "C:\Users\%USERNAME%\miniconda3\python.exe"  set PYTHON_EXE=C:\Users\%USERNAME%\miniconda3\python.exe
if exist "C:\ProgramData\anaconda3\python.exe"         set PYTHON_EXE=C:\ProgramData\anaconda3\python.exe
if exist "C:\anaconda3\python.exe"                     set PYTHON_EXE=C:\anaconda3\python.exe

REM Try PATH-based python
if "%PYTHON_EXE%"=="" (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not "%%i"=="%%i:Microsoft\WindowsApps=%%i" (
            set PYTHON_EXE=%%i
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python not found on this system!
    echo.
    echo Please install Python using ONE of these methods:
    echo.
    echo OPTION 1 (Recommended - Anaconda):
    echo   Download: https://www.anaconda.com/download
    echo   Install with default settings, check 'Add to PATH'
    echo.
    echo OPTION 2 (Python.org):
    echo   Download: https://www.python.org/downloads/
    echo   IMPORTANT: Check 'Add Python to PATH' during install
    echo.
    echo After installing, close this window and run setup.bat again.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found: %PYTHON_EXE%
"%PYTHON_EXE%" --version
echo.

REM ── Step 2: Install packages ──────────────────────────────────
echo [1/3] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip --quiet

echo [2/3] Installing core ML packages...
"%PYTHON_EXE%" -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn fpdf2 joblib --quiet

echo [3/3] Installing XGBoost (optional)...
"%PYTHON_EXE%" -m pip install xgboost --quiet

echo.
echo Verifying installation...
"%PYTHON_EXE%" -c "import pandas, numpy, matplotlib, seaborn, sklearn, imblearn, fpdf; print('[OK] All packages installed!')"
if errorlevel 1 (
    echo [WARN] Some packages may have failed. Try running again.
)

echo.
echo ============================================================
echo  SETUP COMPLETE! Now run the project:
echo ============================================================
echo.
echo   "%PYTHON_EXE%" run_project.py
echo.
echo   OR open Jupyter:
echo   "%PYTHON_EXE%" -m jupyter notebook loan_approval_prediction.ipynb
echo.

REM ── Step 3: Ask if user wants to run now ─────────────────────
set /p RUNOW="Run the project now? (y/n): "
if /i "%RUNOW%"=="y" (
    echo.
    echo Running project...
    "%PYTHON_EXE%" run_project.py
)

pause
