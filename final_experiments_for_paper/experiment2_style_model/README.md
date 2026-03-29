Reproducibility Commands

For this experiment we filter from all players with more than 1K games on TWIC per ~early 2026 for all their games with ELO > 2500 by at least one player, de-duplicate, make sure games last at least five plies, cap each player to max 500 games appearing in final dataset (as target), ensure from each player there's 60% blitz, 30% rapid, 10% classical–if that's not possible we drop the player. We partition the output based on the game state hashed w/ md5. Game metadata is maintained.

### Data Filtering

```sh
python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/build_twic_dataset.py --input ./final_experiments_for_paper/experiment2_style_model/raw_pgns_twic/twic_1k_plus_games.pgn --output-dir ./final_experiments_for_paper/experiment2_style_model/filtered_pgns_twic_1k_plus --sqlite ./final_experiments_for_paper/experiment2_style_model/twic_work_1k_plus.db --min-player-games 1000 --min-elo 2500 --max-games-per-player 500 --min-plies 5 --rebuild-db
```

### Extra Filtering by Player Name (remove top 9 gms) and Processing flattened training / eval data

```sh
python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/process_single_player_features_from_twic_dataset.py --input-dir ./final_experiments_for_paper/experiment2_style_model/filtered_pgns_twic_1k_plus --output-dir ./final_experiments_for_paper/experiment2_style_model/flattened_style_jsonl --min-elo 2500 --eco-bucket-mode family2 --workers 16
```

### Train / Eval Split

We split based on the set of players in a game. We first go over all pairs to get a list of player names.
We then slit the player names into 80% train, 10% eval, 10% test. The eval set will be used for hyperparameter 
tuning since we are doing a material amount of it.

Here's the exact function we use to do the split without losing too much data:

if mover in train_players and opp in train_players:
    return "train"

if mover in eval_players and (opp in train_players or opp in eval_players):
    return "eval"

if mover in test_players:
    return "test"

return None

```bash
python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/split_player_conditioned_jsonl.py --input-dir ./final_experiments_for_paper/experiment2_style_model/flattened_style_jsonl --train-dir ./final_experiments_for_paper/experiment2_style_model/splits/train --eval-dir ./final_experiments_for_paper/experiment2_style_model/splits/eval --test-dir ./final_experiments_for_paper/experiment2_style_model/splits/test --manifest-path ./final_experiments_for_paper/experiment2_style_model/splits/manifest.json --seed 42 --workers 16
```

### Pair Data Generation (there are a few variants of this we try)

#### Variant 1
For positive pairs we only match ones that are in the same phase and game type for the same GM
For negative pairs we only match between same game type, different GMs in the same or different phase. We also match with the same GM and different phases and same game type

```bash
    python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/generate_pairwise_style_data.py \
        --train-dir ./final_experiments_for_paper/experiment2_style_model/splits/train \
        --eval-dir ./final_experiments_for_paper/experiment2_style_model/splits/eval \
        --test-dir ./final_experiments_for_paper/experiment2_style_model/splits/test \
        --train-out-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v1/train \
        --eval-out-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v1/eval \
        --test-out-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v1/test \
        --variant v1 \
        --seed 42 \
        --max-positives-per-anchor 8 \
        --max-negatives-per-anchor 32 \
        --require-positive \
        --require-negative

    python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/ prepare_style_training_cache.py \ 
    --train-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v1/train \ 
    --eval-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v1/eval \ 
    --test-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v1/test \ 
    --out-root ./final_experiments_for_paper/experiment2_style_model/pairs_v1_cached
```

### Variant 2...
For positive pairs we only match ones that are in the same phase and game type for the same GM
For negative pairs we only match between same game type, different GMs in the same or different phase. We do *not* match with the same GM and different phases

```bash
    python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/generate_pairwise_style_data.py \
        --train-dir ./final_experiments_for_paper/experiment2_style_model/splits/train \
        --eval-dir ./final_experiments_for_paper/experiment2_style_model/splits/eval \
        --test-dir ./final_experiments_for_paper/experiment2_style_model/splits/test \
        --train-out-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v1/train \
        --eval-out-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v1/eval \
        --test-out-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v1/test \
        --variant v2 \
        --seed 42 \
        --max-positives-per-anchor 8 \
        --max-negatives-per-anchor 32 \
        --require-positive \
        --require-negative
```

### Stockfish Post-Processing Metadata
The below call requires more than 64 GB of ram to reliably run:

```bash
python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/add_stockfish_post_processing_metadata.py \
    --input-root ./final_experiments_for_paper/experiment2_style_model/splits \
    --engine-path /opt/homebrew/bin/stockfish \
    --depth 18 \
    --multipv 10 \
    --workers 8 \
    --threads-per-worker 2 \
    --hash-mb-per-worker 8192 \
    --overwrite
```

### Training

#### Commands

Single run:

```bash
python -m src.grandmaster_dpo.train.style_embeddings_for_gms.train_style_encoder \
  --study v1_phi1_base
```

Overnight sweep:

```bash
python -m src.grandmaster_dpo.train.style_embeddings_for_gms.run_studies \
  --studies v1_phi0_small v1_phi1_base v2_phi1_base
```

#### Hyper-parameter Tuning Summary

1) Temperature (tau)
0.03, 0.05, 0.07, 0.1, 0.15, 0.2

2) Embedding dimension
128, 256, 512

3) Batch size / negatives per batch
InfoNCE benefits a lot from more negatives
128, 256, 512

4) Positive construction
same GM, nearby era vs same GM across all eras
same GM with similar phase/opening vs completely unrestricted

5) Negative construction
random other GM
hard negatives: other GMs with similar openings / similar engine choices

6) Learning rate
1e-4, 3e-4, 1e-3

7) Sequence/input design (w/ ablations)
i) phi0: 5 boards + move
ii) phi1: 5 boards + move + game_type (blitz / rapid / classical)
iii) phi3: 5 boards + move + game_type + avg phi1 of oponent nearest neighbors from black if no move yet else avg move near their last move state

The hope is i) -> ii) -> iii) all improve.

Stage 1: build the static example dataset

From the PGNs, create one row per usable position-window.

For each row store:

5-board history
played move
GM id
opponent id
time control
phase
opening bucket
maybe engine metadata later, but not required

Also save mapping dictionaries:

player_to_examples
time_control_to_examples
phase_to_examples
opening_to_examples

This can live as:

parquet
Arrow
numpy memmap
LMDB
or sharded pt files

For your machine, parquet + Arrow / numpy memmap is probably enough.

Stage 2: train with random positives + random/semihard negatives

For each anchor:

positive: same GM, maybe same time control
negatives: batch negatives or sampled negatives from other GMs

This gives you a clean baseline.

Stage 3: add hard negative mining

Only after you see baseline learning.

At that point:

run encoder over a large sample or full dataset
save embeddings
build FAISS / ANN index
for each anchor, retrieve:
nearest other-GM examples as hard negatives
nearest same-GM examples as hard positives if you want

Then freeze those mined pairs for the next training round or epoch block.

This means you do not rebuild the index every checkpoint. Instead you rebuild it:

every few epochs
or once per training stage
or once after warmup

That is much more practical
