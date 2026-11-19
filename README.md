# Book Recommendation System — Ensemble & XGBoost Reranking

This repository contains our solution and experimental pipelines for the **Recommender Systems Challenge** at **Politecnico di Milano**.

## Overview & Challenge Description

The goal of the challenge is to build a high-precision recommendation system for an **anonymized catalog of books**, predicting the **top-10 most relevant unseen books** for each target user, evaluated via **MAP@10** (Mean Average Precision at cutoff 10).

The dataset consists of:
- **User Rating Matrix (`URM`, `data_train.csv`):** Implicit user–book interactions across thousands of users and items.
- **Item Content Matrix (`ICM`, `data_ICM_metadata.csv`):** Anonymized book metadata features used for content-based and hybrid graph representations.
- **Target Users (`data_target_users_test.csv`):** Users with varying interaction histories, ranging from cold-start/short-profile readers to highly active users.

---

## Methodology & Architecture

Our approach combines **multi-paradigm model ensembles** with a **two-stage candidate generation and XGBoost learning-to-rank (`XGBRanker`) pipeline**.

### 1. Diverse Base Recommenders
We trained, extended, and tuned a wide set of collaborative, content-based, graph-based, matrix factorization, and neural models (via **Optuna** Bayesian hyperparameter optimization):
- **Graph-Based Random Walk Models:**
  - $\text{RP}^3_\beta$ extended with the Item Content Matrix (`RP3betaRecommenderICM`), modeling a tripartite user–item–feature graph.
  - $\text{P}^3_\alpha$ (`P3alphaRecommender`).
- **Sparse Linear Methods (SLIM):**
  - `SLIM ElasticNet` (with feature selection and multi-threaded training) and Cython-accelerated `SLIM BPR`.
- **Neighborhood Models (KNN):**
  - Collaborative (`ItemKNNCF`), Content-Based (`ItemKNNCBF`), and unified CF+CBF hybrids (`ItemKNN_CFCBF_Hybrid_Recommender`, `HybridRP3KNN`) using asymmetric cosine similarity with **TF-IDF** and **BM25** feature weighting.
- **Matrix Factorization & Deep Learning:**
  - `PureSVD` (latent user and item representations), `IALS` (Implicit Alternating Least Squares), `LightFM` (WARP loss), and `MultVAE` (Variational Autoencoders for collaborative filtering).

### 2. Model Ensembles & Hybridization
To leverage complementary signal across model families and user activity levels, we implemented two main ensemble strategies in [`Recommenders/Hybrid.py`](Recommenders/Hybrid.py):
- **Multi-Model Score Ensembling (`ScoresMultipleHybridRecommender` & `ScoresHybridRecommender`):**
  - Combines normalized prediction scores across heterogeneous recommenders (e.g., $\text{RP}^3_\beta\text{-ICM}$ + `SLIM ElasticNet` + `ItemKNN CF+CBF`) using Optuna-optimized convex weights.
- **User-Profile Stratified Ensembling (`UserWideHybridRecommender`):**
  - Partitions users into **10 interaction-length buckets** based on profile size (`user_hybrid_hyperparameter_search.py`).
  - Assigns dedicated recommenders and hyperparameter configurations tailored to each user group—routing sparse/cold profiles toward graph and content-boosted models while leveraging high-capacity `SLIM ElasticNet` models for active users.

### 3. Two-Stage Pipeline with XGBoost Rerankers (`XGBRanker`)
To move beyond linear score blending, we built a **two-stage learning-to-rank pipeline**:
1. **Stage 1 — Out-of-Fold Candidate Generation ([`run_train_kfold.py`](run_train_kfold.py), [`run_ranker_dataset.py`](run_ranker_dataset.py)):**
   - Recall-optimized base models (`Recall@10` / `Recall@50`), multi-score hybrids, and user-wide hybrids are trained across a **10-fold cross-validation (`KFold`)** split to generate candidate book pools per user without data leakage.
2. **Stage 2 — Pairwise Reranking with `XGBRanker` ([`7-tomaz-suller-xgboost.ipynb`](7-tomaz-suller-xgboost.ipynb)):**
   - An **XGBoost pairwise ranker** (`objective="rank:pairwise"`, `eval_metric="map@10"`) reranks the candidate pool for each user group, using engineered user, item, and ensemble-level features:
     - **Ensemble Scores & Agreement:** Individual model prediction scores (`score_*`), per-model candidate rank, and **recommender agreement** (number of ensemble models recommending the same book).
     - **Cross-Model Score Statistics:** Row-wise mean, standard deviation, minimum, maximum, skewness, and kurtosis across the ensemble predictions.
     - **Item & User Metadata Features:** Item popularity, user profile length (activity), user popularity bias (affinity toward top-10, top-100, and top-1000 popular books), mean item–item content similarity, mean user–user collaborative similarity, and top-5 `PureSVD` user and item latent factors.

---

## Evaluation & Metric Notes

Because the repository contains multiple experimental notebooks using different data splits and evaluation denominators, care must be taken when comparing reported scores across notebooks:

1. **Single Models & Hybrids ([`9-tomaz-suller-submission.ipynb`](9-tomaz-suller-submission.ipynb), [`GabsNotebook.ipynb`](GabsNotebook.ipynb)):**
   - Evaluated via `EvaluatorHoldout` on a **70% train / 10% validation / 20% test** holdout split (`seed=1234`, evaluating on `33,827` users with $\ge 1$ holdout interaction).
   - `EvaluatorHoldout` reports two variants at cutoff $k=10$:
     - `MAP` (dividing by fixed cutoff $k=10$): hybrids reach **~0.0227–0.0229**.
     - `MAP_MIN_DEN` (dividing by $\min(|\text{ground truth}|, k)$): hybrids reach **~0.0557–0.0558** (and up to **0.0594** on individual user-profile buckets).
2. **XGBoost Reranker Metrics ([`7-tomaz-suller-xgboost.ipynb`](7-tomaz-suller-xgboost.ipynb)):**
   - **Internal Training Log (`validation_0-map@10 ≈ 0.60`):** This is XGBoost's internal group-level metric measured *only within each user's candidate rows* prior to item deduplication. It is an internal optimization objective, **not** a catalog-wide recommendation score.
   - **Post-Deduplication Fold Score (`0.0877` in Cell 81):** Computed on **Fold 9 of a 10-fold CV split** (90% train / 10% validation, `random_state=42`) using the $\min(|\text{actual}|, 10)$ denominator (`map_at_k`). Because `decode_targets(X_val, y_val)` constructs the ground-truth dictionary from positive rows inside `X_val`, it evaluates over the **`18,456` users who have at least one relevant item retrieved in their candidate pool** (recall-conditioned reranking MAP@10). Because both the underlying train/validation split (90/10 vs. 70/10/20) and the evaluated user subset differ, `0.0877` is not directly comparable to the full-catalog `EvaluatorHoldout` scores in `GabsNotebook.ipynb` and `9-tomaz-suller-submission.ipynb`.

---

## Notebooks & Key Scripts

All experimental work, model exploration, ensembling, and reranking pipelines can be inspected in the repository's `.ipynb` notebooks and root scripts:

### Core Challenge Notebooks
| Notebook | Primary Author(s) | Description |
| :--- | :--- | :--- |
| [`7-tomaz-suller-xgboost.ipynb`](7-tomaz-suller-xgboost.ipynb) | Tomaz Maia Suller | End-to-end **XGBoost Reranker (`XGBRanker`)** pipeline: 10-fold candidate generation, feature engineering (popularity bias, similarity, ensemble score statistics), pairwise ranking training, feature importance analysis, and Optuna hyperparameter tuning. |
| [`8-tomaz-suller-gabs-code.ipynb`](8-tomaz-suller-gabs-code.ipynb) | Gabriele Bandino & Tomaz Maia Suller | Exploration, evaluation, and Optuna hyperparameter search across `P3alpha`, `RP3beta`, `ItemKNN`, `SLIM ElasticNet`, `MultVAE`, and `LightFM`. |
| [`9-tomaz-suller-submission.ipynb`](9-tomaz-suller-submission.ipynb) | Tomaz Maia Suller | **Ensemble & submission pipeline** combining single models, binary/multi-score hybrids (`ScoresMultipleHybridRecommender`), and user-stratified ensembles (`UserWideHybridRecommender`). |
| [`GabsNotebook.ipynb`](GabsNotebook.ipynb) | Gabriele Bandino | Comprehensive experimentation notebook covering data splitting, collaborative/content-based filtering, neural models (`MultVAE`, `LightFM`), Optuna tuning, and hybrid prototyping. |
| [`Hybrid_RP3_ICM+SlimEN_Recall.ipynb`](Hybrid_RP3_ICM+SlimEN_Recall.ipynb) | Gabriele Bandino | Recall-oriented hybridization of $\text{RP}^3_\beta$ (with ICM) and `SLIM ElasticNet` for candidate generation. |
| [`XGBoostTestREC.ipynb`](XGBoostTestREC.ipynb) & [`notebooks/XGBoostTest.ipynb`](notebooks/XGBoostTest.ipynb) | Gabriele Bandino | Initial **XGBoost Reranker (`XGBRanker`)** prototyping notebooks for candidate extraction, target encoding, sparse feature construction, and pairwise learning-to-rank. |

### Automation & Training Scripts
- [`run_train_kfold.py`](run_train_kfold.py) *(Tomaz Maia Suller)* — Parallelized 10-fold training of recall- and MAP-optimized candidate models (`ItemKNNCF`, `ItemKNN_CFCBF`, `SLIMElasticNet`, `RP3betaICM`, `P3alpha`) and user-stratified hybrids.
- [`run_ranker_dataset.py`](run_ranker_dataset.py) *(Tomaz Maia Suller)* — Constructs the training and submission `.parquet` datasets for `XGBRanker`, computing ensemble agreement, score distribution moments, `PureSVD` latent factors, and user/item similarity features.
- [`user_hybrid_hyperparameter_search.py`](user_hybrid_hyperparameter_search.py) *(Tomaz Maia Suller)* — Tunes recommenders independently across user profile-length groups using Optuna.
- [`run_score_hybrid_optimisation.py`](run_score_hybrid_optimisation.py) *(Tomaz Maia Suller)* — Optimizes blending weights for `ScoresMultipleHybridRecommender`.
- [`run_hyperparameter_search.py`](run_hyperparameter_search.py) *(Tomaz Maia Suller)* — Automated Optuna hyperparameter tuning for individual recommenders.

---

## Authors & Individual Contributions

- **Gabriele Bandino ([@Gabrielebandino](https://github.com/Gabrielebandino))**
  - Designed and implemented the **Tripartite Graph $\text{RP}^3_\beta$ with Item Content Matrix** ([`Recommenders/GraphBased/RP3betaRecommenderICM.py`](Recommenders/GraphBased/RP3betaRecommenderICM.py)), extending graph-based random walks to jointly leverage user–item interactions (`URM`) and book metadata features (`ICM`).
  - Developed custom graph + neighborhood hybrids ([`Recommenders/GabsHybridTesting/HybridRP3KNN.py`](Recommenders/GabsHybridTesting/HybridRP3KNN.py)) and the **Recall@10-optimized $\text{RP}^3_\beta\text{-ICM}$ + `SLIM ElasticNet` hybrid** ([`Hybrid_RP3_ICM+SlimEN_Recall.ipynb`](Hybrid_RP3_ICM+SlimEN_Recall.ipynb)).
  - Led exploratory experimentation and Optuna hyperparameter optimization across collaborative, content-based, matrix factorization, and deep learning recommenders (`MultVAE`, `LightFM`, `SLIM ElasticNet`, `ItemKNN`, `P3alpha`, `RP3beta`) in [`GabsNotebook.ipynb`](GabsNotebook.ipynb) and [`8-tomaz-suller-gabs-code.ipynb`](8-tomaz-suller-gabs-code.ipynb).
  - Prototyped the initial **two-stage `XGBRanker` reranking and feature-engineering pipeline** ([`XGBoostTestREC.ipynb`](XGBoostTestREC.ipynb), [`notebooks/XGBoostTest.ipynb`](notebooks/XGBoostTest.ipynb)).

- **Tomaz Maia Suller ([@tomaz-suller](https://github.com/tomaz-suller))**
  - Architected the competition dataset manager and splitting utilities ([`Data_manager/competition.py`](Data_manager/competition.py)) and parallelized hyperparameter tuning scripts ([`run_hyperparameter_search.py`](run_hyperparameter_search.py), [`user_hybrid_hyperparameter_search.py`](user_hybrid_hyperparameter_search.py), [`run_score_hybrid_optimisation.py`](run_score_hybrid_optimisation.py)).
  - Implemented the modular ensemble framework in [`Recommenders/Hybrid.py`](Recommenders/Hybrid.py) (`ScoresHybridRecommender`, `ScoresMultipleHybridRecommender`, and the 10-bucket user-profile stratified `UserWideHybridRecommender`) and the ensemble submission workflow ([`9-tomaz-suller-submission.ipynb`](9-tomaz-suller-submission.ipynb)).
  - Engineered the **10-fold out-of-fold candidate training pipeline** ([`run_train_kfold.py`](run_train_kfold.py)), the **automated ranker dataset builder & statistical feature extractor** ([`run_ranker_dataset.py`](run_ranker_dataset.py) — ensemble agreement, score distribution moments, `PureSVD` latent factors, and popularity-bias metrics), and the scaled **`XGBRanker` training & Optuna CV pipeline** ([`7-tomaz-suller-xgboost.ipynb`](7-tomaz-suller-xgboost.ipynb)).

---

## Environment & Setup

1. Install dependencies using `uv`, `pip`, or Conda:
   ```bash
   conda env create -f environment.yml
   conda activate RecSysFramework
   ```
2. Compile Cython extensions (used by matrix factorization and SLIM BPR models):
   ```bash
   python run_compile_all_cython.py
   ```

---

## Acknowledgments

Built upon the base Recommender Systems framework developed by [Maurizio Ferrari Dacrema](https://mauriziofd.github.io/) for the *Recommender Systems* course at **Politecnico di Milano**.
