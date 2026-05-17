FROM python:3.11-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY calibre ./calibre
COPY benchmarks ./benchmarks

RUN uv sync --extra cloud --no-dev --frozen

FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin calibre

COPY --from=builder --chown=calibre:calibre /app /app

USER calibre

ENTRYPOINT ["calibre"]
CMD ["health"]
