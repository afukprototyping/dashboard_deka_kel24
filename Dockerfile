FROM python:3.10-slim

WORKDIR /app

# Menghindari error interaksi OS
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# GCP Cloud Run akan menggunakan variable PORT
EXPOSE 8080

# Eksekusi server Streamlit
CMD streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0