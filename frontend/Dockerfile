FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY frontend ./frontend

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
