@echo off
chcp 65001 >nul
setlocal
title Podcast Clipper
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    goto :run
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    goto :run
)

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    goto :run
)

echo ==============================================================
echo   Python が見つかりませんでした。
echo.
echo   Podcast Clipper を実行するには Python 3.10 以上が必要です。
echo   以下からインストールしてください:
echo   https://www.python.org/downloads/
echo.
echo   インストール後、このファイルをもう一度ダブルクリックしてください。
echo ==============================================================
pause
exit /b 1

:run
echo 使用する Python: %PYTHON_EXE%
echo.
set PYTHONUTF8=1
"%PYTHON_EXE%" launch.py

echo.
echo ==============================================================
echo   Podcast Clipper のプロセスは終了しました。
echo   このウィンドウは何かキーを押すと閉じます。
echo ==============================================================
pause >nul
