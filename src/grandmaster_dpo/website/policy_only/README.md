# Policy-Only And Postgame ECS/Fargate Runbook

This folder now contains two related ECS/Fargate deployment targets:

- the live `policy_only` HTTP game service
- the `game_finished_postprocessing` SQS-driven worker service

Together they support:

- serving authoritative live game turns
- publishing finished-game events to SNS
- routing those events through SNS -> SQS
- postprocessing completed games with depth-22 Stockfish analysis
- writing completed game rows and analysis rows into DynamoDB

The live game service still follows the same core serving idea as before:

- one image can contain one or more GM personas,
- one shared code path,
- selected GM-specific policy and timer-head weights baked into the image at build time,
- Stockfish-assisted move selection at request time,
- authoritative game-state responses from the backend.

The postgame worker follows a different model:

- one shared worker image for all finished games
- consumes from an SQS queue subscribed to the finished-game SNS topic
- only deletes messages after DynamoDB write + postgame analysis succeed
- throws on failures so ECS/SQS redrive behavior can retry and eventually DLQ

The intended personas discussed so far are:

- `carlsen`: positional
- `kasparov`: aggressive/classical
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

The finished-game worker lives alongside this package:

```text
website/
  policy_only/
    ...
  game_finished_postprocessing/
    main.py                      SQS long-poll worker entrypoint
    processor.py                 SNS/SQS parsing + Dynamo + Stockfish processing
    Dockerfile                   ECS worker image
```

## End-To-End Architecture

The intended production flow is:

```text
frontend
  -> ALB
  -> ECS/Fargate policy_only service
  -> Redis / Valkey for authoritative in-progress state
  -> when game finishes: SNS topic publish
  -> SNS subscription fanout
  -> SQS queue: garry-chess-games-finished
  -> ECS/Fargate postgame worker service
  -> DynamoDB table: garry-chess-games
```

Recommended AWS resources:

- ALB for the live `policy_only` API
- ECS service for each live GM persona
- SNS topic:
  - `arn:aws:sns:us-east-1:437720536299:garry-chess-games-finished`
- SQS queue subscribed to that topic:
  - `garry-chess-games-finished`
- SQS DLQ:
  - `garry-chess-games-finished-dlq`
- DynamoDB table:
  - `garry-chess-games`
- ElastiCache Valkey for authoritative in-progress state

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

The Docker build accepts a comma- or space-separated `GM_NAMES` build arg. It copies only those selected GM artifacts into the final runtime image. The older single-GM `GM_NAME` arg still works when `GM_NAMES` is omitted.

The build expects a policy checkpoint to exist in one of these locations for each selected GM. The famous-player path is preferred when present, which is currently how Kasparov is packaged:

```text
website_famous_player_experiments/experiment2_style_model/trained_models_single_gm_twic/<gm_name>/policy_best.pt
```

The fallback path is the current paper-experiment layout used by Carlsen, Firouzja, and Praggnanandhaa:


```text
final_experiments_for_paper/experiment2_style_model/trained_models_single_gm_twic/<gm_name>/policy_best_sft_and_dpo_w_style_v3_beta=0.60_dpo_loss_weight=0.60_style_tau=0.25_embedding_model=final_v3_phi1_tau0_25_warm_from_v2final__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42.pt
```

The timer head uses:

```text
processed/single_gm/time_per_move/train_val/<gm_name>/timer_head_best.pt
```

Kasparov currently has a policy checkpoint but no timer head in this layout. The build allows that and copies Carlsen's timer head into the selected GM's runtime model folder as a fallback. Local/runtime model resolution does the same fallback if a GM-specific timer head is missing but Carlsen's exists.

These are copied into:

```text
/opt/models/style_policy/<gm_name>/policy_dpo_best.pt
/opt/models/timer_models/<gm_name>/timer_head_best.pt
```

## Build Args

- `GM_NAMES`: comma- or space-separated GM personas to bake into the image, for example `kasparov,carlsen,firouzja,praggnanandhaa`
- `GM_NAME`: single-GM compatibility arg used only when `GM_NAMES` is omitted
- `STOCKFISH_REF`: which Stockfish git ref to build, default `sf_18`

Because many local development machines are Apple Silicon while ECS/Fargate often runs `linux/amd64`, the build commands below explicitly produce `linux/amd64` images.

## 1. Build And Deploy The Live Policy Game Tasks

This section is for the HTTP ECS services that expose:

- `GET /healthz`
- `POST /games`
- `GET/POST /games/{game_id}/clock*`

### Local Build Commands

Run from repo root.

### Grouped Multi-GM Image

```bash
docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/policy_only/Dockerfile \
  --build-arg GM_NAMES=carlsen,firouzja,praggnanandhaa,kasparov \
  --build-arg STOCKFISH_REF=sf_18 \
  -t garry-chess-policy-only:carlsen-firouzja-pragg-kasparov \
  --load .
```

### Single-GM Compatibility

```bash
docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/policy_only/Dockerfile \
  --build-arg GM_NAME=carlsen \
  --build-arg STOCKFISH_REF=sf_18 \
  -t garry-chess-policy-only:carlsen \
  --load .
```

## Local Run

Run the image:

```bash
docker run --rm -p 8080:8080 garry-chess-policy-only:carlsen-firouzja-pragg-kasparov
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
    "gm_name": "kasparov",
    "bot_id": "kasparov",
    "game_type_id": "gm_kasparov_blitz",
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

### Build, Tag, And Push Multi-GM Policy Image

```bash
docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/policy_only/Dockerfile \
  --build-arg GM_NAMES=carlsen,firouzja,praggnanandhaa,kasparov \
  --build-arg STOCKFISH_REF=sf_18 \
  -t "${ECR_REPO}:carlsen-firouzja-pragg-kasparov" \
  --load .

docker tag "${ECR_REPO}:carlsen-firouzja-pragg-kasparov" "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:carlsen-firouzja-pragg-kasparov"

docker push "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:carlsen-firouzja-pragg-kasparov"
```

### Requesting A GM

For a multi-GM container, pass `gm_name` in the game request. The older `game_type_id` inference remains as a fallback, so `game_type_id=gm_carlsen_blitz` still selects Carlsen if `gm_name` is omitted.

The finished-game SNS payload keeps the existing nested `request` shape for postprocessing; the API-only selector `gm_name` is omitted from `payload["request"]`. The selected GM is still present in `response.analysis.gm_name`.

### Recommended Runtime Environment Variables For Live Policy Tasks

Set these in the ECS task definition:

- `POLICY_ONLY_GM_NAMES`
- `STOCKFISH_THREADS`
- `STOCKFISH_HASH_MB`
- `STOCKFISH_TIMEOUT_S`
- `GAME_STATE_TTL_SECONDS`
- `GAME_STATE_KEY_PREFIX`
- `POLICY_ONLY_GAMES_FINISHED_SNS_TOPIC_ARN`
- `POLICY_ONLY_GAMES_FINISHED_SNS_PUBLISH_ATTEMPTS`
- `POLICY_ONLY_GAMES_FINISHED_SNS_RETRY_BACKOFF_SECONDS`

For Valkey-backed state also set:

- `POLICY_ONLY_REDIS_URL`

The env var keeps the Redis-compatible name because the service uses a Redis protocol client. That works fine with ElastiCache Valkey.

Recommended starting values:

```text
POLICY_ONLY_GM_NAMES=carlsen,firouzja,praggnanandhaa,kasparov
STOCKFISH_THREADS=16
STOCKFISH_HASH_MB=128
STOCKFISH_TIMEOUT_S=20.0
GAME_STATE_TTL_SECONDS=86400
GAME_STATE_KEY_PREFIX=policy_only
POLICY_ONLY_GAMES_FINISHED_SNS_TOPIC_ARN=arn:aws:sns:us-east-1:437720536299:garry-chess-games-finished
POLICY_ONLY_GAMES_FINISHED_SNS_PUBLISH_ATTEMPTS=3
POLICY_ONLY_GAMES_FINISHED_SNS_RETRY_BACKOFF_SECONDS=0.25
```

### Build And Deploy Checklist For Multi-GM Policy Service

Multi-GM example:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=437720536299
export ECR_REPO=garry-chess-policy-only
export GM_NAMES=carlsen,firouzja,praggnanandhaa,kasparov
export IMAGE_TAG=carlsen-firouzja-pragg-kasparov

docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/policy_only/Dockerfile \
  --build-arg GM_NAMES="${GM_NAMES}" \
  --build-arg STOCKFISH_REF=sf_18 \
  -t "${ECR_REPO}:${IMAGE_TAG}" \
  --load .

docker tag "${ECR_REPO}:${IMAGE_TAG}" \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

docker push \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
```

Then in ECS:

1. register / update the task definition revision
2. point the container image at the pushed ECR tag
3. force a new ECS deployment for the service
4. wait for target group health to become healthy
5. verify:

```bash
curl -i https://YOUR_POLICY_API_HOST/healthz
```

## 2. Build And Deploy The Postgame Processing Tasks

This section is for the SQS-driven ECS worker that:

- consumes finished-game events from SQS
- writes completed game rows to DynamoDB
- runs depth-22 Stockfish postgame analysis
- writes `POSTGAME#...` analysis records

### Worker Image Build

Set these first:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=437720536299
export ECR_REPO_POSTGAME=garry-chess-game-finished-postprocessing
```

Create the worker ECR repository if needed:

```bash
aws ecr describe-repositories --repository-names "$ECR_REPO_POSTGAME" --region "$AWS_REGION" >/dev/null 2>&1 || \
aws ecr create-repository --repository-name "$ECR_REPO_POSTGAME" --region "$AWS_REGION"
```

Build, tag, and push:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f src/grandmaster_dpo/website/game_finished_postprocessing/Dockerfile \
  --build-arg STOCKFISH_REF=sf_18 \
  -t "${ECR_REPO_POSTGAME}:latest" \
  --load .

docker tag "${ECR_REPO_POSTGAME}:latest" \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_POSTGAME}:latest"

docker push \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_POSTGAME}:latest"
```

### Worker Runtime Environment Variables

Set these in the ECS worker task definition:

- `GAME_FINISHED_SQS_QUEUE_URL`
- `POLICY_ONLY_GAMES_TABLE_NAME`
- `POSTGAME_ANALYSIS_DEPTH`
- `POSTGAME_STOCKFISH_MULTIPV`
- `POSTGAME_STOCKFISH_THREADS`
- `POSTGAME_STOCKFISH_HASH_MB`
- `POSTGAME_STOCKFISH_TIMEOUT_S`
- `GAME_FINISHED_POSTPROCESSING_LOG_LEVEL`
- `SQS_WAIT_TIME_SECONDS`
- `SQS_VISIBILITY_TIMEOUT_SECONDS`
- `SQS_MAX_NUMBER_OF_MESSAGES`
- `EXIT_ON_IDLE` if you want one-shot ad hoc runs instead of a long-lived service

Recommended starting values:

```text
GAME_FINISHED_SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/437720536299/garry-chess-games-finished
POLICY_ONLY_GAMES_TABLE_NAME=garry-chess-games
POSTGAME_ANALYSIS_DEPTH=22
POSTGAME_STOCKFISH_MULTIPV=5
POSTGAME_STOCKFISH_THREADS=4
POSTGAME_STOCKFISH_HASH_MB=512
POSTGAME_STOCKFISH_TIMEOUT_S=60.0
GAME_FINISHED_POSTPROCESSING_LOG_LEVEL=INFO
SQS_WAIT_TIME_SECONDS=20
SQS_VISIBILITY_TIMEOUT_SECONDS=900
SQS_MAX_NUMBER_OF_MESSAGES=1
EXIT_ON_IDLE=false
```

### Worker IAM Permissions

The worker task role should be allowed to:

- `sqs:ReceiveMessage`
- `sqs:DeleteMessage`
- `sqs:GetQueueAttributes`
- `sqs:ChangeMessageVisibility` (recommended)
- `dynamodb:GetItem`
- `dynamodb:PutItem`
- `dynamodb:UpdateItem`
- `dynamodb:Query`
- `dynamodb:BatchGetItem`
- `logs:CreateLogStream`
- `logs:PutLogEvents`

The live policy service task role should be allowed to:

- `sns:Publish` on
  - `arn:aws:sns:us-east-1:437720536299:garry-chess-games-finished`

## 3. ECS Config For The Live Policy Game Tasks

This is the manual console path for the live HTTP game services.

This is the simplest manual setup path in the AWS console.

### 3.1 Create Or Reuse The ECR Repository

In the AWS console:

1. Open Amazon ECR.
2. Create a private repository named something like `garry-chess-policy-only`.
3. Push the multi-GM policy image to it.

### 3.2 Create An ECS Cluster

1. Open Amazon ECS.
2. Choose `Clusters`.
3. Choose `Create cluster`.
4. Use an ECS cluster that supports Fargate tasks.
5. Give it a name like `garry-chess-policy-only-cluster`.

### 3.3 Create Valkey / ElastiCache First

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

### 3.4 Create A Task Definition

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
   - `POLICY_ONLY_GM_NAMES`
   - `STOCKFISH_THREADS`
   - `STOCKFISH_HASH_MB`
   - `STOCKFISH_TIMEOUT_S`
   - `GAME_STATE_TTL_SECONDS`
   - `GAME_STATE_KEY_PREFIX`
   - `POLICY_ONLY_GAMES_FINISHED_SNS_TOPIC_ARN`
   - `POLICY_ONLY_GAMES_FINISHED_SNS_PUBLISH_ATTEMPTS`
   - `POLICY_ONLY_GAMES_FINISHED_SNS_RETRY_BACKOFF_SECONDS`
7. Add `POLICY_ONLY_REDIS_URL`:
   - either as a plain env var
   - or preferably through ECS container secrets from Secrets Manager / SSM Parameter Store
8. Give the task role permission to publish to the SNS topic.
9. Configure logs to CloudWatch Logs.

Suggested first-pass sizing:

- CPU: `2048`
- memory: `4096` or `8192`

You will likely want to tune this after measuring startup time and move latency.

If using Secrets Manager, create a secret first:

1. Open AWS Secrets Manager.
2. Create a secret containing the Valkey endpoint URL.
3. In the ECS task definition container, add it under `Secrets`.
4. Map it to the env var name `POLICY_ONLY_REDIS_URL`.

### 3.5 Create An Application Load Balancer

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

### 3.6 Create The ECS Service

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

### 3.7 Add Security Groups

Typical setup:

- ALB security group:
  - inbound `80` and/or `443` from the internet
  - outbound allowed to the task security group
- ECS task security group:
  - inbound `8080` from the ALB security group
  - outbound allowed to the cache on `6379`
  - outbound allowed to SNS / AWS public endpoints or NAT route if tasks are in private subnets
- Cache security group:
  - inbound `6379` from the ECS task security group

### 3.8 DNS / TLS Setup

For a clean API setup:

1. request an ACM certificate for the hostname such as `policy.api.garrychess.ai`
2. attach it to the ALB HTTPS listener
3. create a Route 53 alias A record:
   - `policy.api` -> your ALB
4. test:

```bash
curl -i https://policy.api.garrychess.ai/healthz
```

### 3.9 Test The Service

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
    "gm_name": "kasparov",
    "bot_id": "kasparov",
    "game_type_id": "gm_kasparov_blitz",
    "player_color": "white"
  }'
```

### 3.10 Important Scaling Note

If you run more than one ECS task for a persona and still use the current in-memory store, clock/game synchronization will break because each task has its own local memory.

For true multi-task scaling, use a shared backend such as ElastiCache Valkey and keep all `/games` and `/games/{game_id}/clock*` endpoints using that shared store.

With the current code, that Redis-protocol path is now implemented. In ECS, the main thing is wiring `POLICY_ONLY_REDIS_URL` correctly and making sure the cache security group allows inbound `6379` from the ECS task security group.

### 3.11 Suggested Console Wiring Summary

If you want the shortest checklist:

1. Create the ECR repo and push the multi-GM image.
2. Create the Valkey cache in ElastiCache.
3. Copy the cache endpoint.
4. Put the cache URL into Secrets Manager as `POLICY_ONLY_REDIS_URL`.
5. Create the ECS cluster.
6. Create the ECS task definition:
   - image = your ECR image
   - port = `8080`
   - secret/env = `POLICY_ONLY_REDIS_URL`
   - env = `POLICY_ONLY_GM_NAMES`, `GAME_STATE_TTL_SECONDS`, Stockfish envs
7. Create the ALB and target group with `/healthz`.
8. Create the ECS service attached to that target group.
9. Make security groups allow:
   - internet -> ALB `80/443`
   - ALB -> ECS task `8080`
   - ECS task -> cache `6379`

## 4. ECS Config / Service For The Postgame Processing Tasks

This is the manual console path for the SQS-driven worker service.

### 4.1 SNS -> SQS Setup

Create or verify:

1. SNS topic:
   - `arn:aws:sns:us-east-1:437720536299:garry-chess-games-finished`
2. SQS main queue:
   - `garry-chess-games-finished`
3. SQS DLQ:
   - `garry-chess-games-finished-dlq`

On the main SQS queue:

- configure the DLQ redrive policy
- recommended starting `maxReceiveCount`: `10`

On the SNS topic subscription:

- subscribe the main SQS queue to the topic
- use the queue policy that allows SNS to send messages to that queue

Delivery semantics:

- the live game service publishes finished-game events to SNS
- SNS fans out to the SQS queue
- the worker deletes the SQS message only after successful processing
- on failure, it does not delete the message
- SQS retries and eventually redrives to the DLQ

### 4.2 Create The Worker Task Definition

1. In ECS, create a new Fargate-compatible task definition
2. Use the worker image from:
   - `src/grandmaster_dpo/website/game_finished_postprocessing/Dockerfile`
3. Add the worker env vars from the section above
4. Configure CloudWatch logging
5. Give the task role:
   - SQS consume permissions
   - DynamoDB table read/write permissions

Suggested first-pass sizing:

- CPU: `2048`
- memory: `4096`

Increase this if depth-22 analysis latency is too high.

### 4.3 Create The Worker ECS Service

You do not need an ALB for this service.

1. Create a new ECS service from the worker task definition
2. Desired count can start at `0` or `1`
3. Put it in private subnets if you prefer
4. Security group:
   - no inbound needed
   - outbound required to:
     - SQS
     - DynamoDB
     - CloudWatch Logs

### 4.4 Autoscaling For The Worker Service

This worker is a good fit for ECS service autoscaling on SQS backlog.

Recommended pattern:

- min capacity: `0`
- max capacity: some small starting ceiling like `5` or `10`
- scale metric based on SQS visible messages / backlog per task

The important semantic is:

- if queue is empty, it is fine for the service to scale to zero
- if queue grows, ECS should scale workers up

### 4.5 Visibility Timeout And Retries

Set SQS queue visibility timeout long enough for worst-case depth-22 analysis.

Recommended starting point:

- queue visibility timeout: `15 minutes`
- worker env:
  - `SQS_VISIBILITY_TIMEOUT_SECONDS=900`

If analysis can take longer than that, increase both.

If visibility timeout is too short:

- the same message may be delivered again while a worker is still analyzing it
- duplicates are still safe in principle because the DynamoDB writes are idempotent-ish
- but it is noisy and wastes compute

### 4.6 Failure Behavior

The worker intentionally raises on postprocessing failures.

That means:

- success:
  - write completed game to DynamoDB
  - write postgame analysis rows
  - delete SQS message
- failure:
  - log the exception
  - do not delete the SQS message
  - let SQS retry
  - eventually redrive to `garry-chess-games-finished-dlq`

This is the intended failure mode because it makes alarms and debugging much easier.

### 4.7 Worker Verification

Useful checks after deployment:

1. CloudWatch logs for the worker container
2. SQS queue depth:
   - visible messages
   - in-flight messages
3. DLQ count:
   - should normally stay at zero
4. DynamoDB rows:
   - parent `GAME`
   - `INFERENCE#...`
   - `POSTGAME#BOARD_STATE#DEPTH#22#...`
   - `POSTGAME#MOVE#DEPTH#22#...`
   - `POSTGAME#SUMMARY#DEPTH#22#...`

## Notes On The Finished-Game Event Payload

The postgame worker expects the finished-game SNS/SQS payload to carry a full game snapshot, not just a final result.

That event now includes enough for downstream analysis:

- `start_fen`
- `final_fen`
- `moves_compact`
- `inference_positions`
- actor metadata
- final game status
- clocks and timing context

That lets the worker:

- write the `GAME` item first
- write inference child items
- then do full postgame engine analysis

without having to reconstruct the whole game externally.

## Why This Structure Helps Later

When you add more service endpoints for related games or richer clock APIs, the folder layout already supports it:

- add a schema file or extend `schemas/game.py`
- add a new endpoint file under `api/routes/`
- add corresponding orchestration logic in `service/game_service.py` or a sibling service module
- include it in `api/router.py`

That avoids a future return to one giant handler file.
