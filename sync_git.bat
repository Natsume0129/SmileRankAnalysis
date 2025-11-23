@echo off
REM === 如果脚本不在仓库根目录，把下面这一行改成你的仓库路径 ===
REM cd /d "C:\path\to\your\repo"

echo ===== Auto Git Sync =====

REM 检查是否是 git 仓库
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo 当前目录不是 Git 仓库，脚本结束。
    pause
    exit /b 1
)

REM 检查是否有改动（含未跟踪文件）
git status --porcelain | findstr /r /c:".*" >nul 2>&1
if errorlevel 1 (
    echo 没有需要提交的改动。
    pause
    exit /b 0
)

REM 生成一个简单的自动提交信息（带日期时间）
for /f "tokens=1-4 delims=/:. " %%a in ("%date% %time%") do (
    set y=%%d
    set m=%%b
    set d=%%c
    set hh=%%a
)
set msg=auto-commit %y%-%m%-%d% %time%

echo 有改动，开始提交...
git add -A
git commit -m "%msg%"
if errorlevel 1 (
    echo git commit 失败，请检查错误信息。
    pause
    exit /b 1
)

echo 推送到远程仓库...
git push
if errorlevel 1 (
    echo git push 失败，请检查网络或权限。
    pause
    exit /b 1
)

echo 完成。
pause
