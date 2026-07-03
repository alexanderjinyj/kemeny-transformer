# Archive

Superseded files kept for reference. Nothing in here is used by the current code.

- `backup_train_ddp.py` — old backup of the training script (now `scripts/train.py`)
- `train_ddp_refactored.py` — abandoned refactor that `exec()`d the original script
- `compute_borda_footrule_200_250.py` / `compute_decor_distances_200_250.py` —
  hardcoded 200/250-item copies; replaced by the `--items` flag on
  `scripts/evaluation/compute_borda_footrule.py` and `compute_decor_distances.py`
- `decor_random_test.R` / `decor_repeat_test.R` / `decor_jiggling_test.R` —
  three near-identical scripts; replaced by the parameterized `r_scripts/decor_test.R`
- `preflib_university_duplicate/` — byte-identical copy of `results/preflib_university/`
  that used to live under `test_dataset/`
- `nation_rankings_root_copy.csv` — diverged copy of `data/nation/nation_rankings.csv`
  that used to sit at the repo root
- `setup.py` — replaced by `pyproject.toml`
- `mallow_keyword_report.json`, `test_result_completeness.json` — one-off analysis reports
