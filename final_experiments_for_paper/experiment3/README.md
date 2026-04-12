# Reproducibility Guide


To run the stockfish studies, run 

```sh
python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py --gm_name carlsen --split val --sf_depth 5 --sf_tops 10
```

```sh
for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py --gm_name $gm --split val --sf_depth 10,8,6,4,2 --sf_tops 10; done && for gm in carlsen nakamura firouzja caruana wei vincent giri gukesh praggnanandhaa; do python ./src/grandmaster_dpo/eval/single_gm/stockfish_tree_search/eval_all_types_in_one.py --gm_name $gm --split val --sf_depth 16 --sf_tops 10; done
```

There's a slight naming inconsistency for the sf_cache. In particular, restrict cp window is redundant and the files produced are agnostic of it to support more downstream consumers (Eval runners)