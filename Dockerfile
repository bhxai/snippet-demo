# syntax=docker/dockerfile:1

# Stage 1: Build Frontend (Node)
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
# FIXED: Use full npm ci (includes devDeps like Vite for build)
RUN npm ci
COPY frontend/ ./
# Set for prod: Relative API path (assumes fetch('/api/...') in code)
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build -- --base=/static/  # Outputs to dist/ (update vite.config.js outDir if not default)

# Stage 2: Backend (Python) + Serve Frontend Static
FROM python:3.11-slim AS production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

WORKDIR /app

# Install Python deps
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Copy backend code
COPY backend/ /app/backend/

# Copy built frontend to backend's static dir (create if needed)
RUN mkdir -p /app/backend/static
COPY --from=frontend-build /app/frontend/dist /app/backend/static

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
