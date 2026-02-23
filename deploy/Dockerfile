# ============================================================
# GitHub MCP Server — Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder): install dependencies + pre-download ML models
#   Stage 2 (runtime): lean image with only what's needed to run
# ============================================================

# ---- Stage 1: Builder ----------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for git and compiling some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (layer cache)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download ML models so the runtime image starts without network calls.
# Models are cached in HuggingFace's default cache dir.
ENV HF_HOME=/hf_cache
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer, CrossEncoder
print("Downloading nomic-embed-text-v1.5 ...")
SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
print("Downloading cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Models downloaded successfully.")
EOF


# ---- Stage 2: Runtime ----------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy pre-downloaded HuggingFace models
ENV HF_HOME=/hf_cache
COPY --from=builder /hf_cache /hf_cache

# Copy application source
COPY src/ ./src/

# Data directories (will be mounted as Docker volumes in production)
RUN mkdir -p /data/repo /data/index

# Non-root user for security
RUN useradd --system --uid 1001 --gid 0 mcp && \
    chown -R mcp:0 /app /data /hf_cache
USER mcp

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the server
CMD ["python", "-m", "uvicorn", "src.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--log-level", "info", "--no-access-log"]
