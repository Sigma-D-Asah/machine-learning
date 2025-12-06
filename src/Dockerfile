# Use official Python 3.13 slim image
FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /src

# Install OS deps required for some packages and tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user to run the app
RUN adduser --disabled-password --gecos '' --uid 1000 appuser \
    && mkdir -p /home/appuser \
    && chown -R appuser:appuser /home/appuser

# Copy only requirements and install them to leverage Docker cache
COPY requirements.txt /src/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /src/requirements.txt

# Copy application files
COPY . /src

# Make sure files are owned by the appuser
RUN chown -R appuser:appuser /src

USER appuser

# Expose uvicorn default port
EXPOSE 8001

# Healthcheck for Uvicorn FastAPI
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8001/api/v1/failure/health || exit 1

# Start Uvicorn server
ENTRYPOINT ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8001"]
