Reproducibility Commands


For each of the 9 GMs in the twic raw pgn folder (i.e. carlsen):


1) Filter to 300 blitz games (60%), 150 rapid games (30%), and 50 tournament games (10%). 
Also min plies, max plies of each are 20, 500 respectively. We avoid game types like chess 960 and other variants.

For Unix:
```bash
python ./src/grandmaster_dpo/data_processing/single_gm/clean_and_filter_initial_png.py  --pgn ./final_experiments_for_paper/experiment1/raw_pgns_twic/carlsen.pgn --gm_name carlsen --out_dir ./final_experiments_for_paper/experiment1/cleaned_and_filtered_pgns_twic
```

For Windows:

```bash
python .\src\grandmaster_dpo\data_processing\single_gm\clean_and_filter_initial_png.py --pgn .\final_experiments_for_paper\experiment1\raw_pgns_twic\carlsen.pgn --gm_name carlsen --out_dir .\final_experiments_for_paper\experiment1\cleaned_and_filtered_pgns_twic
```

2) Split data into 80% train/val, 20% test

```bash
python ./src/grandmaster_dpo/data_processing/single_gm/split_cleaned_into_train_val.py --gm_name carlsen --in_dir ./final_experiments_for_paper/experiment1/cleaned_and_filtered_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/train_val_pgns_twic
```

```bash
python .\src\grandmaster_dpo\data_processing\single_gm\split_cleaned_into_train_val.py --gm_name carlsen --in_dir .\final_experiments_for_paper\experiment1\cleaned_and_filtered_pgns_twic --out_dir .\final_experiments_for_paper\experiment1\train_val_pgns_twic
```

3) Take train / test splits and for each generate pairs of chosen vs a maia2 best alternative rejected move. Includes metadata about the move like the opening move of the game / game id. For the chosen / alt move includes the maia2 chosen / rejected probabilities. The win_prob is the win_prob maia2 gives from the given position before the chosen move is applied. It's independent of the move probs.


```bash
python ./src/grandmaster_dpo/data_processing/single_gm/z_curate_dpo_from_train_val.py --gm_name carlsen --split_dir ./final_experiments_for_paper/experiment1/train_val_pgns_twic
```

```bash
python .\src\grandmaster_dpo\data_processing\single_gm\z_curate_dpo_from_train_val.py --gm_name carlsen --split_dir .\final_experiments_for_paper\experiment1\train_val_pgns_twic --sf_path D:\jcarl\stockfish\stockfish-windows-x86-64-avx2.exe --sf_threads 32
```

4) Train a DPO model:


```bash
python ./src/grandmaster_dpo/train/single_gm/train_dpo_maia2.py --gm_name carlsen --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic --use_kl_penalty
python ./src/grandmaster_dpo/train/single_gm/train_dpo_maia2.py --gm_name carlsen --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic
```

```bash
python .\src\grandmaster_dpo\train\single_gm\train_dpo_maia2.py --gm_name carlsen --train_val_folder .\final_experiments_for_paper\experiment1\train_val_pgns_twic --out_dir .\final_experiments_for_paper\experiment1\trained_models_twic --use_kl_penalty
python .\src\grandmaster_dpo\train\single_gm\train_dpo_maia2.py --gm_name carlsen --train_val_folder .\final_experiments_for_paper\experiment1\train_val_pgns_twic --out_dir .\final_experiments_for_paper\experiment1\trained_models_twic
```

5) Train SFT model:

```bash
python ./src/grandmaster_dpo/train/single_gm/train_sft_maia2.py --gm_name carlsen --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic
```

```bash
python .\src\grandmaster_dpo\train\single_gm\train_sft_maia2.py --gm_name carlsen --train_val_folder .\final_experiments_for_paper\experiment1\train_val_pgns_twic --out_dir .\final_experiments_for_paper\experiment1\trained_models_twic
```

6) Train SFT pairwise model:


```bash
python ./src/grandmaster_dpo/train/single_gm/train_sft_pairwise_maia2.py --gm_name carlsen --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic
```

7) Eval DPO model:


```bash
python .\src\grandmaster_dpo\eval\single_gm\eval_dpo_maia_single_gm.py --gm_name carlsen --train_val_folder .\final_experiments_for_paper\experiment1\train_val_pgns_twic --out_dir .\final_experiments_for_paper\experiment1\eval_results_twic --model_dir .\final_experiments_for_paper\experiment1\trained_models_twic --used_kl_penalty
```


8) Eval SFT Model:

```bash
python .\src\grandmaster_dpo\eval\single_gm\eval_sft_maia_single_gm.py --gm_name carlsen --train_val_folder .\final_experiments_for_paper\experiment1\train_val_pgns_twic --out_dir .\final_experiments_for_paper\experiment1\eval_results_twic --model_dir .\final_experiments_for_paper\experiment1\trained_models_twic
```

9) Eval SFT Pairwise Model:

```bash
python .\src\grandmaster_dpo\eval\single_gm\eval_sft_pairwise_maia_single_gm.py --gm_name carlsen --train_val_folder .\final_experiments_for_paper\experiment1\train_val_pgns_twic --out_dir .\final_experiments_for_paper\experiment1\eval_results_twic --model_dir .\final_experiments_for_paper\experiment1\trained_models_twic
```
