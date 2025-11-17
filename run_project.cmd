python scripts/preprocess_wimans.py --dataset_root training_dataset --band 5.0 --target_T 3000 --max_users 5 --use_wavelet_pp --out_dir training_dataset/processed_5g_pp

python scripts/build_splits.py --metadata_csv training_dataset/processed_5g_pp/metadata.csv --out_csv training_dataset/processed_5g_pp/splits_5g_pp.csv --train_envs classroom empty_room --val_envs classroom empty_room --test_envs meeting_room --val_ratio 0.2

python scripts/check_pipeline.py --config configs/wimans_5g_pp.yaml

python scripts/train_sadu.py --config configs/wimans_5g_pp.yaml

python scripts/train_baseline_cnn.py --config configs/wimans_5g_pp_baseline.yaml

python scripts/eval_sadu.py --config configs/wimans_5g_pp.yaml --per_env

python scripts/infer_demo.py --config configs/wimans_5g_pp.yaml --sample_id sample_1

python scripts/run_cross_env_experiments.py --base_config_sadu configs/wimans_5g_pp.yaml --base_config_baseline configs/wimans_5g_pp_baseline.yaml --metadata_csv training_dataset/processed_5g_pp/metadata.csv --processed_dir training_dataset/processed_5g_pp --envs meeting_room
