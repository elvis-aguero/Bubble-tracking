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

REM Resolve a real bash.exe (Git Bash / WSL bash in PATH / manual install)
set "BASH_EXE="
where bash >nul 2>nul
if %ERRORLEVEL% EQU 0 set "BASH_EXE=bash"

if not defined BASH_EXE if exist "%ProgramFiles%\Git\bin\bash.exe" set "BASH_EXE=%ProgramFiles%\Git\bin\bash.exe"
if not defined BASH_EXE if exist "%ProgramW6432%\Git\bin\bash.exe" set "BASH_EXE=%ProgramW6432%\Git\bin\bash.exe"
if not defined BASH_EXE if exist "%ProgramFiles(x86)%\Git\bin\bash.exe" set "BASH_EXE=%ProgramFiles(x86)%\Git\bin\bash.exe"
if not defined BASH_EXE if exist "%LocalAppData%\Programs\Git\bin\bash.exe" set "BASH_EXE=%LocalAppData%\Programs\Git\bin\bash.exe"

if defined BASH_EXE (
    echo [1/2] Found bash. Running launcher script...
    "%BASH_EXE%" "%SOURCE_DIR%bubbly_flows\scripts\xanylabel.sh"
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

echo ERROR: Could not find a compatible launcher runtime.
echo - Preferred: Install Git for Windows (bash.exe), then re-run this file.
echo - Fallback: Install Python and create x-labeling-env as described in USER_GUIDE.md.
pause

:end
pause
