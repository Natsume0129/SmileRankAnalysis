@echo off
REM ===== Auto Git Push =====

REM (Optional) If the script is NOT in repo root, specify your repo path:
REM cd /d "C:\path\to\your\repo"

echo ===== Auto Git Push =====

REM Check if current directory is a git repo
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo Not a git repository. Script aborted.
    pause
    exit /b 1
)

REM Check if there are any changes
git status --porcelain | findstr /r /c:".*" >nul 2>&1
if errorlevel 1 (
    echo No changes to commit.
    pause
    exit /b 0
)

REM Generate English auto commit message with timestamp
for /f "tokens=1-4 delims=/:. " %%a in ("%date% %time%") do (
    set year=%%d
    set month=%%b
    set day=%%c
    set hour=%%a
)
set msg=auto-commit %year%-%month%-%day% %time%

echo Changes detected. Committing...
git add -A
git commit -m "%msg%"
if errorlevel 1 (
    echo git commit failed.
    pause
    exit /b 1
)

echo Pushing to remote...
git push
if errorlevel 1 (
    echo git push failed.
    pause
    exit /b 1
)

echo Done.
pause
