@echo off
echo ============================================
echo EUVlitho Streamlit Interactive Demo
echo ============================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10 or higher.
    pause
    exit /b 1
)
echo.

echo Installing/Upgrading dependencies...
pip install -r streamlit_requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo.

echo Starting Streamlit application...
echo.
echo Once the app starts, it will open in your browser automatically.
echo If not, open: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo ============================================
echo.

streamlit run streamlit_app.py

pause
