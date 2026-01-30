@echo off
REM Description: Double-click launcher for Bubbly Flows X-AnyLabeling on Windows.

SET SOURCE_DIR=%~dp0
cd /d "%SOURCE_DIR%"

echo ===================================================
echo     BUBBLY FLOWS - X-AnyLabeling Launcher
echo ===================================================

REM Check if bubbly_flows exists
if not exist "bubbly_flows" (
    echo ERROR: 'bubbly_flows' directory not found in: %SOURCE_DIR%
    echo Make sure this script is in the root of the Bubbly-tracking repository.
    pause
    exit /b 1
)

REM Try to find bash (sh.exe) which usually comes with Git for Windows
where sh >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [1/2] Found bash/sh. Running script via bash...
    sh bubbly_flows/scripts/xanylabel.sh
    goto end
)

REM If no bash, try to run it via Python if the user has it
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [1/2] Bash not found. Attempting to start via Python activation...
    if exist "x-labeling-env\Scripts\activate.bat" (
        call x-labeling-env\Scripts\activate.bat
        xanylabeling
    ) else (
        echo ERROR: Virtual environment not found (x-labeling-env).
        echo Please follow the User Guide to set up the environment first.
        pause
    )
    goto end
)

echo ERROR: Neither 'bash' (sh.exe) nor 'python' was found in your PATH.
echo Please install Git for Windows (includes bash) or Python to use this launcher.
pause

:end
pause
