FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy the app code
COPY . /app

# Create a user (non-root)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

ENTRYPOINT ["bash", "-lc"]
# Default command will be overridden per-service in docker-compose