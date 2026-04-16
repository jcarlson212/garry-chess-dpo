# Reproducibility Guide


To run the stockfish studies, run 

```sh
python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py --gm_name carlsen --split val --sf_depth 5 --sf_tops 10
```

```sh
for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py --gm_name $gm --split val --sf_depth 10,8,6,4,2 --sf_tops 10; done
```

```sh
for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py --gm_name $gm --split val --sf_depth 12 --sf_tops 10; done
```

```sh
for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py --gm_name $gm --split val --sf_depth 14 --sf_tops 10; done
```

This just generates reference data. It's hardcoded to for depth 16 to run on just 500 rows.
```sh
for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py --gm_name $gm --split val --sf_depth 16 --sf_tops 10; done
```

Above runs were canceled after generating sf eval cache for stockfish. Then ran this end-to-end (sf eval cache picked up)
note: some runs ran with temperature = 1.0 by default even though it isn't in the signature
```sh
for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do
  echo "Running $gm..."

  if ! python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py \
    --gm_name $gm --split val --sf_depth 2,4,6,8,10,12 --sf_tops 10; then

    echo "❌ Failed: $gm" >> failed_gms.txt
  else
    echo "✅ Success: $gm"
  fi
done

for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do
  echo "Running $gm..."

  if ! python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py \
    --gm_name $gm --split val --sf_depth 2,4,6,8,10,12 --sf_tops 10 --use_gibbs; then

    echo "❌ Failed: $gm" >> failed_gms.txt
  else
    echo "✅ Success: $gm"
  fi
done

for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do
  echo "Running $gm..."

  if ! python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py \
    --gm_name $gm --split val --sf_depth 2,4,6,8,10,12 --sf_tops 10 --use_gibbs --temperature 0.25; then

    echo "❌ Failed: $gm" >> failed_gms.txt
  else
    echo "✅ Success: $gm"
  fi
done

for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do
  echo "Running $gm..."

  if ! python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py \
    --gm_name $gm --split val --sf_depth 2,4,6,8,10,12 --sf_tops 10 --use_gibbs --temperature 0.50; then

    echo "❌ Failed: $gm" >> failed_gms.txt
  else
    echo "✅ Success: $gm"
  fi
done

for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do
  echo "Running $gm..."

  if ! python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py \
    --gm_name $gm --split val --sf_depth 2,4,6,8,10,12 --sf_tops 10 --use_gibbs --temperature 0.75; then

    echo "❌ Failed: $gm" >> failed_gms.txt
  else
    echo "✅ Success: $gm"
  fi
done

for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do
  echo "Running $gm..."

  if ! python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py \
    --gm_name $gm --split val --sf_depth 2,4,6,8,10,12 --sf_tops 10 --use_gibbs --temperature 1.50; then

    echo "❌ Failed: $gm" >> failed_gms.txt
  else
    echo "✅ Success: $gm"
  fi
done
```

There's a slight naming inconsistency for the sf_cache. In particular, restrict cp window is redundant and the files produced are agnostic of it to support more downstream consumers (Eval runners)