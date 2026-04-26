# Dockerfile for the policy-only HTTP API running on ECS/Fargate.

# ---------- Stage 1: Build Stockfish ----------
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


# ---------- Stage 2: Select model artifacts ----------
FROM python:3.11-slim AS model_artifacts

ARG GM_NAME=carlsen
ARG GM_NAMES=""

WORKDIR /artifacts

COPY ./final_experiments_for_paper/experiment2_style_model/trained_models_single_gm_twic/ /artifacts/style_policy_src/
COPY ./website_famous_player_experiments/experiment2_style_model/trained_models_single_gm_twic/ /artifacts/famous_style_policy_src/
COPY ./processed/single_gm/time_per_move/train_val/ /artifacts/timer_src/

RUN set -eux; \
    selected_gms="${GM_NAMES:-$GM_NAME}"; \
    for gm in $(echo "$selected_gms" | tr ',' ' '); do \
      mkdir -p "/out/models/style_policy/${gm}" "/out/models/timer_models/${gm}"; \
      famous_policy="/artifacts/famous_style_policy_src/${gm}/policy_best.pt"; \
      paper_policy="/artifacts/style_policy_src/${gm}/policy_best_sft_and_dpo_w_style_v3_beta=0.60_dpo_loss_weight=0.60_style_tau=0.25_embedding_model=final_v3_phi1_tau0_25_warm_from_v2final__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42.pt"; \
      if [ -f "$famous_policy" ]; then \
        cp "$famous_policy" "/out/models/style_policy/${gm}/policy_dpo_best.pt"; \
      else \
        cp "$paper_policy" "/out/models/style_policy/${gm}/policy_dpo_best.pt"; \
      fi; \
      timer_head="/artifacts/timer_src/${gm}/timer_head_best.pt"; \
      fallback_timer_head="/artifacts/timer_src/carlsen/timer_head_best.pt"; \
      if [ -f "$timer_head" ]; then \
        cp "$timer_head" "/out/models/timer_models/${gm}/timer_head_best.pt"; \
      else \
        echo "No timer head for ${gm}; using carlsen timer head fallback."; \
        cp "$fallback_timer_head" "/out/models/timer_models/${gm}/timer_head_best.pt"; \
      fi; \
    done


# ---------- Stage 3: Runtime image ----------
FROM python:3.11-slim

ARG GM_NAME=carlsen
ARG GM_NAMES=""

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV STOCKFISH_PATH=/opt/bin/stockfish
ENV MODEL_ROOT=/opt/models
ENV MAIA_DEVICE=cpu
ENV STOCKFISH_THREADS=16
ENV STOCKFISH_HASH_MB=128
ENV STOCKFISH_TIMEOUT_S=20.0
ENV POLICY_ONLY_GM_NAME=${GM_NAME}
ENV POLICY_ONLY_GM_NAMES=${GM_NAMES}
ENV PUZZLE_SCENARIOS_PATH=/opt/puzzles/scenario_miner_output.jsonl

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ make libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/bin
COPY --from=stockfish_stage /out/stockfish /opt/bin/stockfish
RUN chmod +x /opt/bin/stockfish

WORKDIR /app

RUN mkdir -p /opt/puzzles
COPY --from=model_artifacts /out/models/ /opt/models/
COPY ./website_data_mining_experiments/scenario_miner_output.jsonl /opt/puzzles/scenario_miner_output.jsonl

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

EXPOSE 8080

CMD ["uvicorn", "grandmaster_dpo.website.policy_only.app:app", "--host", "0.0.0.0", "--port", "8080"]
