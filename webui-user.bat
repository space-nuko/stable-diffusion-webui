@echo off

set PYTHON=C:\Users\yuno\AppData\Local\Programs\Python\Python310\python.exe
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--deepdanbooru --xformers --opt-split-attention --listen --enable-insecure-extension-access --enable-batching

call webui.bat
