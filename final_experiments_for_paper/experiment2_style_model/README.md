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

    python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/prepare_style_training_cache.py \ 
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

    python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/prepare_style_training_cache.py \ 
        --train-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v2/train \ 
        --eval-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v2/eval \ 
        --test-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v2/test \ 
        --out-root ./final_experiments_for_paper/experiment2_style_model/pairs_v2_cached
```

### Variant 3

We took variant 2 and filtered it to just the 8 negatives that are furthest away. This dataset is used to further tune an already trained model on variant 2 to get it even better on harder negatives / upweight them.

```bash
python ./src/grandmaster_dpo/data_processing/style_embeddings_for_gms/prepare_style_training_cache_v3_hardneg.py \
    --in-root ./final_experiments_for_paper/experiment2_style_model/pairs_v2_cached \
    --out-root ./final_experiments_for_paper/experiment2_style_model/pairs_v3_cached \
    --hard-negatives-per-pair 8 \
    --embedding-batch-size 4096
```

The model used for determining hard vs soft negatives was `screen_v1_phi0_tau0_75__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.75__seed-42`.

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

### Eval

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir ./final_experiments_for_paper/experiment2_style_model/trained_models/screen_v1_phi0_tau0_75__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.75__seed-42 \
--pairs-dir ./final_experiments_for_paper/experiment2_style_model/pairs_v1 \
--output-dir ./final_experiments_for_paper/experiment2_style_model/eval_outputs/screen_v1_phi0_tau0_75 \
--checkpoint-name best \
--save-embeddings
```
