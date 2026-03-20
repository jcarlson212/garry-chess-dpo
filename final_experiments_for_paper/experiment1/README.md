Reproducibility Commands

Note that experiment 1 encompasses all style experiments in the initial paper submitted to IEEE CoG. Followup revisions and/or arxiv versions will have separate experiments for a new depth study (how depth saturates style), a model for the style fidelity rather than a heuristic, and styleometry.

For each of the 9 GMs in the twic raw pgn folder (i.e. carlsen) -- we don't provide these PGNs as they were paid for --:


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

4) Train NLL + DPO Models


```bash
foreach ($gm in "nakamura","firouzja","carlsen","wei","vincent","giri","gukesh","praggnanandhaa","caruana") { python ./src/grandmaster_dpo/train/single_gm/train_sft_and_dpo_maia2.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic }
```

5) Train NLL models:

```bash
foreach ($gm in "nakamura","firouzja","carlsen","wei","vincent","giri","gukesh","praggnanandhaa","caruana") { python ./src/grandmaster_dpo/train/single_gm/train_sft_maia2.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic }
```

6) Train NLL pairwise model:


```bash
foreach ($gm in "carlsen","wei","vincent","giri","gukesh","nakamura","caruana","praggnanandhaa","firouzja") { python ./src/grandmaster_dpo/train/single_gm/train_sft_pairwise_maia2.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic }
```


7) Train NLL + DPO re-weighted for style v1 models:

run the below for style_tau = 0.25, 0.75, 1.25, dpo_loss_weight = 0.1, 0.2, 0.4
```bash
foreach ($gm in "caruana","nakamura","firouzja","carlsen","wei","vincent","giri","gukesh","praggnanandhaa") { python ./src/grandmaster_dpo/train/single_gm/train_sft_and_dpo_w_style_sim_utility_weight_maia2.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic --dpo_loss_weight 0.4 --style_tau 0.75 }
```

8) Train NLL + DPO re-weighted for style v2 models:

run the below for style_tau = 0.25, 0.75, 1.25, dpo_loss_weight = 0.1, 0.2, 0.4
```bash
foreach ($gm in "caruana","nakamura","firouzja","carlsen","wei","vincent","giri","gukesh","praggnanandhaa") { python ./src/grandmaster_dpo/train/single_gm/train_sft_and_dpo_w_style_v2_maia2.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic --dpo_loss_weight 0.4 --style_tau 1.25 }
```


9) Train DPO Models

```bash
foreach ($gm in "nakamura","firouzja","wei","vincent","giri","gukesh","praggnanandhaa","caruana") { python ./src/grandmaster_dpo/train/single_gm/train_dpo_maia2.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/trained_models_twic --betas 0.02 0.05 0.1 0.2 0.4 0.6 }
```


10) Eval NLL + DPO Models


```bash
foreach ($gm in "nakamura","firouzja","wei","vincent","giri","gukesh","praggnanandhaa","caruana") { python ./src/grandmaster_dpo/eval/single_gm/eval_sft_and_dpo_maia_single_gm.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/eval_results_twic --model_dir ./final_experiments_for_paper/experiment1/trained_models_twic }
```

11) Eval NLL models:

```bash
foreach ($gm in "nakamura","firouzja","carlsen","wei","vincent","giri","gukesh","praggnanandhaa","caruana") { python .\src\grandmaster_dpo\eval\single_gm\eval_sft_maia_single_gm.py --gm_name $gm --train_val_folder .\final_experiments_for_paper\experiment1\train_val_pgns_twic --out_dir .\final_experiments_for_paper\experiment1\eval_results_twic --model_dir .\final_experiments_for_paper\experiment1\trained_models_twic }
```


12) Eval NLL pairwise model:

```bash
foreach ($gm in "nakamura","firouzja","carlsen","wei","vincent","giri","gukesh","praggnanandhaa","caruana") { python .\src\grandmaster_dpo\eval\single_gm\eval_sft_pairwise_maia_single_gm.py --gm_name $gm --train_val_folder .\final_experiments_for_paper\experiment1\train_val_pgns_twic --out_dir .\final_experiments_for_paper\experiment1\eval_results_twic --model_dir .\final_experiments_for_paper\experiment1\trained_models_twic }
```


13) Eval Base Maia2 model:

```bash
foreach ($gm in "nakamura","firouzja","carlsen","wei","vincent","giri","gukesh","praggnanandhaa","caruana") { python ./src/grandmaster_dpo/eval/single_gm/eval_no_op_maia_single_gm.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/eval_results_twic }
```

14) Eval NLL + DPO re-weighted for style v1 models

```bash
foreach ($gm in "nakamura","firouzja","carlsen","wei","vincent","giri","gukesh","praggnanandhaa","caruana") { python ./src/grandmaster_dpo/eval/single_gm/eval_sft_and_dpo_w_style_sim_utility_weight_maia2.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/eval_results_twic --model_dir ./final_experiments_for_paper/experiment1/trained_models_twic }
```

15) Eval NLL + DPO re-weighted for style v2 models:

```bash
foreach ($gm in "nakamura","firouzja","carlsen","wei","vincent","giri","gukesh","praggnanandhaa","caruana") { python ./src/grandmaster_dpo/eval/single_gm/eval_sft_and_dpo_w_style_v2_maia_single_gm.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/eval_results_twic --model_dir ./final_experiments_for_paper/experiment1/trained_models_twic }
```

16) Eval DPO Models

```bash
foreach ($gm in "nakamura","firouzja","carlsen","wei","vincent","giri","gukesh","praggnanandhaa","caruana") { python ./src/grandmaster_dpo/eval/single_gm/eval_dpo_maia_single_gm.py --gm_name $gm --train_val_folder ./final_experiments_for_paper/experiment1/train_val_pgns_twic --out_dir ./final_experiments_for_paper/experiment1/eval_results_twic --model_dir ./final_experiments_for_paper/experiment1/trained_models_twic }
```

17) Generate paper graphs

```bash
python .\src\grandmaster_dpo\graphs\generate_paper_figures.py --eval_root ./final_experiments_for_paper/experiment1/eval_results_twic --out_dir ./final_experiments_for_paper/experiment1/paper_figures
```

