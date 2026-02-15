# Dockerfile (Lambda container image, Python 3.11) with Stockfish + baked GM .pt models

# ---------- Stage 1: Get Stockfish on Amazon Linux ----------
FROM amazonlinux:2 AS stockfish_stage
RUN yum -y update && \
    yum -y install git gcc gcc-c++ make which && \
    yum clean all

WORKDIR /build

# You can pin a version/tag if you want reproducibility (recommended).
# Example tags: sf_16, sf_17, etc. If you don't care, you can use master.
ARG STOCKFISH_REF=sf_16
RUN git clone https://github.com/official-stockfish/Stockfish.git && \
    cd Stockfish && \
    git checkout ${STOCKFISH_REF}

# Build for modern x86_64 (works fine for Lambda x86_64)
RUN cd Stockfish/src && \
    make -j"$(nproc)" profile-build && \
    strip stockfish

# Put the binary somewhere predictable for COPY
RUN mkdir -p /out && cp /build/Stockfish/src/stockfish /out/stockfish

# ---------- Stage 2: Lambda Python 3.11 runtime ----------
FROM public.ecr.aws/lambda/python:3.11

# If any python deps compile wheels
RUN yum install -y gcc gcc-c++ make && yum clean all

# Copy Stockfish binary into /opt/bin
RUN mkdir -p /opt/bin
COPY --from=stockfish_stage /out/stockfish /opt/bin/stockfish
RUN chmod +x /opt/bin/stockfish

# Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# ---- Bake models into the image ----
# Expect repo has:
#   models/<gm_name>/policy_dpo_best.pt
# This will end up at /opt/models/<gm_name>/policy_dpo_best.pt
RUN mkdir -p /opt/models
COPY ./processed/single_gm/train_val/carlsen/policy_dpo_best.pt /opt/models/carlsen/policy_dpo_best.pt
COPY ./processed/single_gm/train_val/nakamura/policy_dpo_best.pt /opt/models/nakamura/policy_dpo_best.pt
COPY ./processed/single_gm/train_val/caruana/policy_dpo_best.pt /opt/models/caruana/policy_dpo_best.pt
COPY ./processed/single_gm/train_val/firouzja/policy_dpo_best.pt /opt/models/firouzja/policy_dpo_best.pt
COPY ./processed/single_gm/train_val/giri/policy_dpo_best.pt /opt/models/giri/policy_dpo_best.pt
COPY ./processed/single_gm/train_val/gukesh/policy_dpo_best.pt /opt/models/gukesh/policy_dpo_best.pt
COPY ./processed/single_gm/train_val/praggnanandhaa/policy_dpo_best.pt /opt/models/praggnanandhaa/policy_dpo_best.pt
COPY ./processed/single_gm/train_val/vincent/policy_dpo_best.pt /opt/models/vincent/policy_dpo_best.pt
COPY ./processed/single_gm/train_val/wei/policy_dpo_best.pt /opt/models/wei/policy_dpo_best.pt

COPY ./processed/single_gm/famous_chessgames_gms/alekhine/policy_best_dpo.pt /opt/models/alekhine/policy_best_dpo.pt
COPY ./processed/single_gm/famous_chessgames_gms/anand/policy_best_dpo.pt /opt/models/anand/policy_best_dpo.pt
COPY ./processed/single_gm/famous_chessgames_gms/botvinnik/policy_best_dpo.pt /opt/models/botvinnik/policy_best_dpo.pt
COPY ./processed/single_gm/famous_chessgames_gms/capablanca/policy_best_dpo.pt /opt/models/capablanca/policy_best_dpo.pt
COPY ./processed/single_gm/famous_chessgames_gms/fischer/policy_best_dpo.pt /opt/models/fischer/policy_best_dpo.pt
COPY ./processed/single_gm/famous_chessgames_gms/kasparov/policy_best_dpo.pt /opt/models/kasparov/policy_best_dpo.pt
COPY ./processed/single_gm/famous_chessgames_gms/lasker/policy_best_dpo.pt /opt/models/lasker/policy_best_dpo.pt
COPY ./processed/single_gm/famous_chessgames_gms/polgar/policy_best_dpo.pt /opt/models/polgar/policy_best_dpo.pt
COPY ./processed/single_gm/famous_chessgames_gms/tal/policy_best_dpo.pt /opt/models/tal/policy_best_dpo.pt

# ---- Environment defaults (override in Lambda config if desired) ----
ENV STOCKFISH_PATH=/opt/bin/stockfish
ENV MODEL_ROOT=/opt/models
ENV MAIA_DEVICE=cpu
ENV STOCKFISH_THREADS=8
ENV STOCKFISH_HASH_MB=128
ENV STOCKFISH_TIMEOUT_S=5.0


# Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy only packaging metadata first (better caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# IMPORTANT:
# - Remove "tkinter" from dependencies OR this will fail.
# - Also ensure "maia2" is resolvable (pip-installable or vendored).
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Now copy the actual source code

COPY  ./src/grandmaster_dpo/website/lambda_handler.py ./
COPY ./maia2_models/ ./maia2_models/

CMD ["lambda_handler.lambda_handler"]
