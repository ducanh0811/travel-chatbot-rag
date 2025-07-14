@echo off
cd /d E:\CDIO2\travel-chatbot-rag

REM 1. Khởi động FastAPI (port 8000)
start "FastAPI" python -m uvicorn main:app --host 127.0.0.1 --port 8000

REM 2. Chờ 3 giây để FastAPI khởi động
timeout /t 3 >nul

REM 3. Chạy ngrok (port 8000) với đường dẫn mới
start "ngrok" "E:\CDIO2\ngrok.exe" http 8000
