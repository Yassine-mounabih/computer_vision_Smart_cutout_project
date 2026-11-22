@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto error
%PY% scripts\download_pretrained_u2netp.py
if errorlevel 1 goto error
echo Installation terminee. Lancez app.py.
pause
exit /b 0
:error
echo Installation echouee. Verifiez votre connexion et les messages ci-dessus.
pause
exit /b 1
