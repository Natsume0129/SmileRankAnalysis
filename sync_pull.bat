@echo off
echo ===== Auto Git Pull (Force Sync from Remote) =====

REM (Optional) If the script is not in repo root, set your repo path:
REM cd /d "C:\path\to\your\repo"

REM Check if current directory is a git repo
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo Not a git repository. Script aborted.
    pause
    exit /b 1
)

REM Check if there are uncommitted changes
git status --porcelain | findstr /r /c:".*" >nul 2>&1
if not errorlevel 1 (
    echo Uncommitted changes detected.
    echo Creating backup before overwrite...

    set ts=%date:~0,4%-%date:~5,2%-%date:~8,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%
    set ts=%ts: =0%

    mkdir backup >nul 2>&1
    mkdir backup\backup_%ts%
    xcopy * backup\backup_%ts%\ /E /H /C /I >nul

    echo Backup created at backup\backup_%ts%
)

echo Fetching remote...
git fetch --all

echo Resetting local to origin/main...
git reset --hard origin/main

echo Done.
pause
