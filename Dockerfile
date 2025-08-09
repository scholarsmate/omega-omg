# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime dependencies first (leverages Docker layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional dev/test tools if you need them inside the image
# COPY requirements-dev.txt ./
# RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source
COPY dsl ./dsl
COPY omg.py ./
COPY highlighter.py ./
COPY README.md LICENSE ./

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default to showing CLI help
ENTRYPOINT ["python", "omg.py"]
CMD ["--help"]
