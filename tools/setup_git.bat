@echo off
cd ..
echo Setting up Git repository for Scripts folder...
echo.

REM Initialize Git repository
git init
echo.

REM Configure Git for the project
echo Please enter your name for Git commits:
set /p GIT_USER_NAME=
git config user.name "%GIT_USER_NAME%"

echo Please enter your email for Git commits:
set /p GIT_USER_EMAIL=
git config user.email "%GIT_USER_EMAIL%"

REM Add all files to Git
git add .

REM Create initial commit
git commit -m "Initial commit of GNSS-InSAR alignment scripts"

echo.
echo Git repository initialized with initial commit.
echo See tools\VERSION_CONTROL.md for usage instructions.
pause
