# Dockerfile for the game-finished postprocessing ECS task.

FROM python:3.11-slim AS stockfish_stage

ARG STOCKFISH_REF=sf_18

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone https://github.com/official-stockfish/Stockfish.git && \
    cd Stockfish && \
    git checkout ${STOCKFISH_REF}

RUN cd Stockfish/src && \
    make -j"$(nproc)" profile-build && \
    strip stockfish

RUN mkdir -p /out && cp /build/Stockfish/src/stockfish /out/stockfish


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STOCKFISH_PATH=/opt/bin/stockfish
ENV POSTGAME_ANALYSIS_DEPTH=22
ENV POSTGAME_STOCKFISH_THREADS=4
ENV POSTGAME_STOCKFISH_HASH_MB=512
ENV POSTGAME_STOCKFISH_TIMEOUT_S=60.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/bin
COPY --from=stockfish_stage /out/stockfish /opt/bin/stockfish
RUN chmod +x /opt/bin/stockfish

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

CMD ["python", "-m", "grandmaster_dpo.website.game_finished_postprocessing.main"]
