FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY data ./data

# Pre-train the risk model at build time so containers start instantly
RUN cd backend && python -c "from app.ml.risk_model import get_bundle; get_bundle()"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
