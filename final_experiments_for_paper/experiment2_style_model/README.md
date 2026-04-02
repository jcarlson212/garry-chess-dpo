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

#### Embeddings
```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v2_phi1_tau0_10_if_winner__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v2 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v2_phi1_tau0_10_if_winner__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v2_phi1_tau0_25_if_winner__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v2 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v2_phi1_tau0_25_if_winner__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v2_phi0_tau0_25_if_winner__pair-v2__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v2 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v2_phi0_tau0_25_if_winner__pair-v2__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v1_phi0_tau0_05__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.05__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v1 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v1_phi0_tau0_05__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.05__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v1_phi1_tau0_05__pair-v1__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.05__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v1 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v1_phi1_tau0_05__pair-v1__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.05__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v1_phi1_tau0_10__pair-v1__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v1 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v1_phi1_tau0_10__pair-v1__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v1_phi1_tau0_25__pair-v1__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v1 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v1_phi1_tau0_25__pair-v1__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/screen_v2_phi0_tau0_10__pair-v2__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v2 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/screen_v2_phi0_tau0_10__pair-v2__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/screen_v2_phi0_tau0_25__pair-v2__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v2 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/screen_v2_phi0_tau0_25__pair-v2__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/screen_v2_phi1_tau0_10__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v2 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/screen_v2_phi1_tau0_10__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.10__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/screen_v2_phi1_tau0_25__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v2 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/screen_v2_phi1_tau0_25__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v1_phi0_tau0_10__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v1 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v1_phi0_tau0_10__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```



```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v1_phi0_tau0_25__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v1 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v1_phi0_tau0_25__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```



--- Once ready...



final_v3_phi1_tau0_10_warm_from_v2final
final_v3_phi1_tau0_25_warm_from_v2final
super_v2_phi1_tau0_25_if_winner
super_v3_phi1_tau0_25_warm_from_v2final

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v1_phi0_tau0_10__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v1 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v1_phi0_tau0_10__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.1__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/final_v1_phi0_tau0_25__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v1 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/final_v1_phi0_tau0_25__pair-v1__phi-phi0__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

```bash
python ./src/grandmaster_dpo/eval/style_embeddings_for_gms/eval_style_embedding_model.py \
--model-dir final_experiments_for_paper/experiment2_style_model/trained_models/super_v3_phi1_tau0_25__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--pairs-dir final_experiments_for_paper/experiment2_style_model/pairs_v2 \
--output-dir final_experiments_for_paper/experiment2_style_model/eval_outputs/super_v3_phi1_tau0_25__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42 \
--splits eval test \
--sampled-embedding-max-players 500 \
--sampled-embedding-max-examples-per-player 16 \
--sampled-embedding-min-examples-per-player 2
```

#### Single GM

```bash
for gm in caruana nakamura firouzja carlsen wei vincent giri gukesh praggnanandhaa; do python ./src/grandmaster_dpo/eval/single_gm/eval_sft_and_dpo_w_style_v3_maia_single_gm.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment2_style_model/eval_results_single_gm_twic --model_dir ./final_experiments_for_paper/experiment2_style_model/trained_models_single_gm_twic; done
```

#### Training Per GM Style Models

Train NLL + DPO re-weighted for style v3 models:


run the below for style_tau = 0.25, 0.75, 1.25, dpo_loss_weight = 0.1, 0.2, 0.4

For initial model to test with:
```bash
for w in 0.1 0.2 0.4 0.6 0.8 1.0; do for tau in 0.25 0.75 1.25; do for gm in caruana nakamura firouzja carlsen wei vincent giri gukesh praggnanandhaa; do python ./src/grandmaster_dpo/train/single_gm/train_sft_and_dpo_w_style_v3_maia2.py --gm_name "$gm" --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment2_style_model/trained_models_single_gm_twic --dpo_loss_weight "$w" --style_tau "$tau" --beta 0.6 --style_embedding_model_checkpoint ./final_experiments_for_paper/experiment2_style_model/trained_models/final_v2_phi1_tau0_25_if_winner__pair-v2__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42/best.pt; done; done; done
```

For intermediate v3 model:

```bash
for w in 0.1 0.2 0.4 0.6 0.8 1.0; do for tau in 0.25 0.75 1.25; do for gm in caruana nakamura firouzja carlsen wei vincent giri gukesh praggnanandhaa; do python ./src/grandmaster_dpo/train/single_gm/train_sft_and_dpo_w_style_v3_maia2.py --gm_name "$gm" --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment2_style_model/trained_models_single_gm_twic --dpo_loss_weight "$w" --style_tau "$tau" --beta 0.6 --style_embedding_model_checkpoint ./final_experiments_for_paper/experiment2_style_model/trained_models/final_v3_phi1_tau0_25_warm_from_v2final__pair-v3__phi-phi1__edim-256__bs-4096__lr-0.0003__tau-0.25__seed-42/best.pt; done; done; done
```


### Graphs

```bash
python src/grandmaster_dpo/graphs/generate_style_embedding_paper_figures.py \
--eval-runs-root final_experiments_for_paper/experiment2_style_model/eval_outputs \
--training-summary-dir final_experiments_for_paper/experiment2_style_model/training_summary \
--output-dir final_experiments_for_paper/experiment2_style_model/paper_plots_v2 \
--split test \
--include-appendix
```

For the per gm models:

```bash
python ./src/grandmaster_dpo/graphs/generate_style_embedding_paper_figures.py --eval_root ./final_experiments_for_paper/experiment1/eval_results_twic/ --v3_eval_root ./final_experiments_for_paper/experiment2_style_model/eval_results_single_gm_twic --out_dir ./final_experiments_for_paper/experiment2_style_model/eval_graphs_twic_single_gm/
```