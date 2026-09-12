# X-LocVul Makefile for Artifact Evaluation

data-prep:
	python src/data_audit/download_linevul.py
	python src/data_audit/data_audit.py
	python src/data_audit/prepare_chrono_data.py

train-e1:
	python src/e1/models/e1_1_codebert.py
	python src/e1/models/e1_2_unixcoder.py
	python src/e1/models/e1_3_unix_multi.py
	python src/e1/models/e1_4_hard_mining.py
	python src/e1/models/e1_5_shuffled_cwe.py
	python src/e1/models/e1_6_hierarchical_cwe.py

train-e3:
	python src/e3/e3_train_locator.py

train-e5:
	python src/e5/e5_train_detector.py

e1:
	python src/e1/compute_e1_metrics.py

e1-bootstrap:
	python src/e1/bootstrap_e1.py
	python src/e1/e1_bootstrap_all.py

e2:
	python src/e2/e2_ablation_mtl.py

e3:
	python src/e3/compute_e3_metrics.py
	python src/e3/compute_e3_average.py
	python src/e3/e3_localization_ablation.py

e4:
	python src/e4/e4_sample_linevul.py
	python src/e4/e4_sample_cases.py
	python src/e4/e4_human_eval.py
	python src/e4/analyze_e4_ratings.py

e5:
	python src/e5/compute_e5_diagnostic.py

e6:
	python src/e6/compute_e6_cost.py

run-pipeline:
	python src/e4/run_full_pipeline.py
