# ---- Stage 1: build the React/TypeScript web app
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- Stage 2: Python API serving the built UI
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GLUCOLAB_DATA_DIR=/data
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY data ./data
COPY --from=web /web/dist ./web/dist

# Pre-train the risk model so containers start instantly
RUN cd backend && python -c "from app.ml.risk_model import get_bundle; get_bundle()"

RUN useradd --create-home appuser && mkdir -p /data && chown -R appuser /app /data
USER appuser
VOLUME ["/data"]
EXPOSE 8000
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
