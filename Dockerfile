FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY calibre ./calibre
COPY benchmarks ./benchmarks

RUN uv sync --extra cloud --extra ml --extra neural --extra ray --no-dev --frozen \
    && find /app/.venv -type f -name "*.pyc" -delete \
    && find /app/.venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /app/.venv/lib/python3.11/site-packages -type d \( -name tests -o -name test \) -prune -exec rm -rf {} + \
    && find /app/.venv/lib/python3.11/site-packages -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cuh" \) -delete \
    && find /app/.venv/lib/python3.11/site-packages -type f \( -name "*.so" -o -name "*.so.*" \) -exec strip --strip-unneeded {} +

FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin calibre

COPY --from=builder --chown=calibre:calibre /app /app

RUN chown calibre:calibre /app

USER calibre

ENTRYPOINT ["calibre"]
CMD ["health"]
