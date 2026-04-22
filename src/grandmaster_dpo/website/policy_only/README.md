# Policy-Only ECS/Fargate Service

This folder now contains a small HTTP service designed for ECS/Fargate deployment.

The service still follows the same core serving idea as before:

- one image per GM persona,
- one shared code path,
- GM-specific policy and timer-head weights baked into the image at build time,
- Stockfish-assisted move selection at request time,
- authoritative game-state responses from the backend.

The intended personas discussed so far are:

- `carlsen`: positional
- `firouzja`: tactical
- `praggnanandhaa`: balanced

## Folder Structure

The service is split so future endpoints can grow cleanly.

```text
policy_only/
  app.py                         FastAPI app
  lambda_handler.py              thin Lambda-compat wrapper
  Dockerfile                     ECS/Fargate container image
  api/
    router.py                    top-level router assembly
    dependencies.py              shared singleton service wiring
    routes/
      games.py                   POST /games
      clocks.py                  clock read/update/sync endpoints
      health.py                  GET /healthz
  schemas/
    game.py                      Pydantic request/response models
    health.py                    Pydantic health model
  service/
    runtime.py                   model loading, timer head, Stockfish logic
    game_service.py              request handling and orchestration
    state.py                     game-state store abstraction
```

The idea is that each endpoint gets its own route file, while orchestration stays in the `service/` layer and schema shape stays explicit in `schemas/`.

## Current Endpoints

- `GET /healthz`
- `POST /games`
- `GET /games/{game_id}/clock`
- `POST /games/{game_id}/clock`
- `POST /games/{game_id}/clock/sync`

The clock endpoints are included now because they are likely to be useful once ECS tasks need to manage related game state around a single GM model.

## Current State Store

The service now supports two backends:

- `RedisGameStateStore` when `POLICY_ONLY_REDIS_URL`, `REDIS_URL`, or `ELASTICACHE_REDIS_URL` is set
- `InMemoryGameStateStore` otherwise

The in-memory store is still useful for:

- local testing
- one-task demos
- early bring-up before Valkey exists

But for real ECS deployment, ElastiCache Valkey is the intended backend.

Even the in-memory fallback now has guardrails:

- TTL controlled by `GAME_STATE_TTL_SECONDS`
- max number of games controlled by `IN_MEMORY_MAX_GAMES`

That prevents unbounded growth during local testing, but it still is not appropriate for multi-task production serving.

## Model Artifacts Expected By The Docker Build

The Docker build expects these files to exist in the repo:

Policy checkpoint:

```text
final_experiments_for_paper/experiment2_style_model/trained_models_single_gm_twic/<gm_name>/policy_best_sft_and_dpo_w_style_v3_beta=0.60_dpo_loss_weight=0.60_style_tau=0.25_embedding_model=final_v3_phi1_tau0_25_warm_from_v2final__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42.pt
```

Timer head:

```text
processed/single_gm/time_per_move/train_val/<gm_name>/timer_head_best.pt
```

These are copied into:

```text
/opt/models/style_policy/<gm_name>/policy_dpo_best.pt
/opt/models/timer_models/<gm_name>/timer_head_best.pt
```

## Build Args

- `GM_NAME`: which GM persona to bake into the image
- `STOCKFISH_REF`: which Stockfish git ref to build, default `sf_18`

Because many local development machines are Apple Silicon while ECS/Fargate often runs `linux/amd64`, the build commands below explicitly produce `linux/amd64` images.

## Local Build Commands

Run from repo root.

### Carlsen

```bash
docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/policy_only/Dockerfile \
  --build-arg GM_NAME=carlsen \
  --build-arg STOCKFISH_REF=sf_18 \
  -t garry-chess-policy-only:carlsen \
  --load .
```

### Firouzja

```bash
docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/policy_only/Dockerfile \
  --build-arg GM_NAME=firouzja \
  --build-arg STOCKFISH_REF=sf_18 \
  -t garry-chess-policy-only:firouzja \
  --load .
```

### Praggnanandhaa

```bash
docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/policy_only/Dockerfile \
  --build-arg GM_NAME=praggnanandhaa \
  --build-arg STOCKFISH_REF=sf_18 \
  -t garry-chess-policy-only:praggnanandhaa \
  --load .
```

## Local Run

Run one image at a time:

```bash
docker run --rm -p 8080:8080 garry-chess-policy-only:carlsen
```

If you changed dependencies locally, refresh your env first:

```bash
pip install -e .
```

Health check:

```bash
curl http://localhost:8080/healthz
```

Example move request:

```bash
curl -X POST http://localhost:8080/games \
  -H 'Content-Type: application/json' \
  -d '{
    "game_id": "test-game-1",
    "client_ply": 0,
    "pre_move_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "client_uci": "e2e4",
    "bot_id": "carlsen",
    "game_type_id": "gm_carlsen_blitz",
    "player_color": "white",
    "clock": {
      "white_ms": 300000,
      "black_ms": 300000
    },
    "timing": {
      "player_move_elapsed_ms": 1200
    },
    "engine_config": {
      "random_seed": 7,
      "use_timer_head": true,
      "use_gibbs": false,
      "cp_gap_window": 60,
      "stockfish_multipv_topk": 10
    }
  }'
```

Read the current authoritative clock state:

```bash
curl http://localhost:8080/games/test-game-1/clock
```

Sync the client against server state:

```bash
curl -X POST http://localhost:8080/games/test-game-1/clock/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "client_ply": 2,
    "client_fen": "some-fen-if-you-want-to-compare"
  }'
```

## ECR Commands

Set these first:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=437720536299
export ECR_REPO=garry-chess-policy-only
```

Create the private ECR repository if needed:

```bash
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 || \
aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION"
```

Authenticate Docker:

```bash
aws ecr get-login-password --region "$AWS_REGION" | \
docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
```

### Build, Tag, And Push Carlsen

```bash
docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/policy_only/Dockerfile \
  --build-arg GM_NAME=carlsen \
  --build-arg STOCKFISH_REF=sf_18 \
  -t "${ECR_REPO}:carlsen" \
  --load .

docker tag "${ECR_REPO}:carlsen" "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:carlsen"

docker push "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:carlsen"
```

### Build, Tag, And Push Firouzja

```bash
docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/policy_only/Dockerfile \
  --build-arg GM_NAME=firouzja \
  --build-arg STOCKFISH_REF=sf_18 \
  -t "${ECR_REPO}:firouzja" \
  --load .

docker tag "${ECR_REPO}:firouzja" "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:firouzja"

docker push "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:firouzja"
```

### Build, Tag, And Push Praggnanandhaa

```bash
docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/policy_only/Dockerfile \
  --build-arg GM_NAME=praggnanandhaa \
  --build-arg STOCKFISH_REF=sf_18 \
  -t "${ECR_REPO}:praggnanandhaa" \
  --load .

docker tag "${ECR_REPO}:praggnanandhaa" "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:praggnanandhaa"

docker push "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:praggnanandhaa"
```

## Recommended Runtime Environment Variables

Set these in the ECS task definition:

- `POLICY_ONLY_GM_NAME`
- `STOCKFISH_THREADS`
- `STOCKFISH_HASH_MB`
- `STOCKFISH_TIMEOUT_S`
- `GAME_STATE_TTL_SECONDS`
- `GAME_STATE_KEY_PREFIX`

For Valkey-backed state also set:

- `POLICY_ONLY_REDIS_URL`

The env var keeps the Redis-compatible name because the service uses a Redis protocol client. That works fine with ElastiCache Valkey.

Recommended starting values:

```text
POLICY_ONLY_GM_NAME=carlsen
STOCKFISH_THREADS=16
STOCKFISH_HASH_MB=128
STOCKFISH_TIMEOUT_S=20.0
GAME_STATE_TTL_SECONDS=86400
GAME_STATE_KEY_PREFIX=policy_only
```

## ECS/Fargate Console Setup

This is the simplest manual setup path in the AWS console.

### 1. Create Or Reuse The ECR Repository

In the AWS console:

1. Open Amazon ECR.
2. Create a private repository named something like `garry-chess-policy-only`.
3. Push one of the persona images to it.

### 2. Create An ECS Cluster

1. Open Amazon ECS.
2. Choose `Clusters`.
3. Choose `Create cluster`.
4. Use an ECS cluster that supports Fargate tasks.
5. Give it a name like `garry-chess-policy-only-cluster`.

### 3. Create Valkey / ElastiCache First

Do this before the ECS service if you want durable shared game state.

Simplest console path:

1. Open Amazon ElastiCache.
2. Choose the Valkey-capable cache path if it is available in your account/region.
3. Create a Valkey cache.
4. For a simple first deployment, choose the serverless Valkey path if available.
5. Put it in the same VPC you plan to use for ECS.
6. Choose or create a security group for the cache.
7. After creation, copy the cache endpoint.

Security group wiring:

- Cache security group inbound:
  - allow TCP `6379`
  - source = the ECS task security group

You can either:

- use no auth if you are keeping this entirely private inside your VPC and your setup is simple, or
- store the cache URL in AWS Secrets Manager and inject it into ECS as a secret

Example URL shape:

```text
redis://YOUR_REDIS_ENDPOINT:6379/0
```

### 4. Create A Task Definition

1. In ECS, choose `Task definitions`.
2. Choose `Create new task definition`.
3. Choose the Fargate-compatible path.
4. Use:
   - operating system: Linux
   - CPU and memory sized for your model workload
   - task execution role: let the console create or choose the standard ECS task execution role
5. Add one container:
   - image URI: your ECR image tag
   - container port: `8080`
   - essential: yes
6. Add environment variables:
   - `POLICY_ONLY_GM_NAME`
   - `STOCKFISH_THREADS`
   - `STOCKFISH_HASH_MB`
   - `STOCKFISH_TIMEOUT_S`
   - `GAME_STATE_TTL_SECONDS`
   - `GAME_STATE_KEY_PREFIX`
7. Add `POLICY_ONLY_REDIS_URL`:
   - either as a plain env var
   - or preferably through ECS container secrets from Secrets Manager / SSM Parameter Store
8. Configure logs to CloudWatch Logs.

Suggested first-pass sizing:

- CPU: `2048`
- memory: `4096` or `8192`

You will likely want to tune this after measuring startup time and move latency.

If using Secrets Manager, create a secret first:

1. Open AWS Secrets Manager.
2. Create a secret containing the Valkey endpoint URL.
3. In the ECS task definition container, add it under `Secrets`.
4. Map it to the env var name `POLICY_ONLY_REDIS_URL`.

### 5. Create An Application Load Balancer

Use an ALB if you want a normal HTTP endpoint.

1. Open EC2 > Load Balancers.
2. Create an `Application Load Balancer`.
3. Make it internet-facing if this is a public API.
4. Pick at least two subnets across AZs.
5. Add a listener on `HTTP : 80` or `HTTPS : 443`.
6. Create a target group:
   - target type: `ip`
   - protocol: HTTP
   - port: `8080`
   - health check path: `/healthz`

For Fargate with `awsvpc`, use `ip` target type, not `instance`.

### 6. Create The ECS Service

1. Open your ECS cluster.
2. Choose `Create service`.
3. Launch type / capacity: Fargate.
4. Choose the task definition revision you just created.
5. Choose desired task count, typically `1` to start.
6. Use networking with:
   - the VPC you want
   - subnets for the tasks
   - a security group that allows traffic from the ALB to port `8080`
7. Attach the ALB and target group:
   - container name: your one service container
   - container port: `8080`
   - target group: the one with `/healthz`

### 7. Add Security Groups

Typical setup:

- ALB security group:
  - inbound `80` and/or `443` from the internet
  - outbound allowed to the task security group
- ECS task security group:
  - inbound `8080` from the ALB security group
  - outbound allowed to the cache on `6379`
- Cache security group:
  - inbound `6379` from the ECS task security group

### 8. Test The Service

Once tasks are healthy behind the ALB:

```bash
curl http://YOUR_ALB_DNS_NAME/healthz
```

and then:

```bash
curl -X POST http://YOUR_ALB_DNS_NAME/games \
  -H 'Content-Type: application/json' \
  -d '{
    "game_id": "prod-test-1",
    "client_ply": 0,
    "pre_move_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "client_uci": "e2e4",
    "bot_id": "carlsen",
    "game_type_id": "gm_carlsen_blitz",
    "player_color": "white"
  }'
```

## Important Scaling Note

If you run more than one ECS task for a persona and still use the current in-memory store, clock/game synchronization will break because each task has its own local memory.

For true multi-task scaling, use a shared backend such as ElastiCache Valkey and keep all `/games` and `/games/{game_id}/clock*` endpoints using that shared store.

With the current code, that Redis-protocol path is now implemented. In ECS, the main thing is wiring `POLICY_ONLY_REDIS_URL` correctly and making sure the cache security group allows inbound `6379` from the ECS task security group.

## Suggested Console Wiring Summary

If you want the shortest checklist:

1. Create the ECR repo and push the GM image.
2. Create the Valkey cache in ElastiCache.
3. Copy the cache endpoint.
4. Put the cache URL into Secrets Manager as `POLICY_ONLY_REDIS_URL`.
5. Create the ECS cluster.
6. Create the ECS task definition:
   - image = your ECR image
   - port = `8080`
   - secret/env = `POLICY_ONLY_REDIS_URL`
   - env = `POLICY_ONLY_GM_NAME`, `GAME_STATE_TTL_SECONDS`, Stockfish envs
7. Create the ALB and target group with `/healthz`.
8. Create the ECS service attached to that target group.
9. Make security groups allow:
   - internet -> ALB `80/443`
   - ALB -> ECS task `8080`
   - ECS task -> cache `6379`

## Why This Structure Helps Later

When you add more service endpoints for related games or richer clock APIs, the folder layout already supports it:

- add a schema file or extend `schemas/game.py`
- add a new endpoint file under `api/routes/`
- add corresponding orchestration logic in `service/game_service.py` or a sibling service module
- include it in `api/router.py`

That avoids a future return to one giant handler file.
