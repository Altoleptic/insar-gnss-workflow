@echo off
echo GNSS-InSAR Analysis Tools Menu
echo =============================
echo.
echo 1. Run workflow (master.py)
echo 2. Run workflow with automatic backup
echo 3. Create a backup of your scripts
echo 4. Restore from a previous backup
echo 5. Set up Git version control
echo 6. Exit
echo.

set /p choice=Enter your choice (1-6): 

if "%choice%"=="1" (
    python master.py
    pause
) else if "%choice%"=="2" (
    call run_workflow_with_backup.bat
) else if "%choice%"=="3" (
    call tools\create_backup.bat
    pause
) else if "%choice%"=="4" (
    call tools\restore_backup.bat
) else if "%choice%"=="5" (
    call tools\setup_git.bat
) else if "%choice%"=="6" (
    exit /b 0
) else (
    echo Invalid choice. Please try again.
    pause
)
