@echo off
REM Double-click to start the app (run setup first - see README.md)
call venv\Scripts\activate
streamlit run app.py
pause
