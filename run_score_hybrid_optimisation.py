import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import optuna
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.model_selection import KFold

from Data_manager.competition import load, load_raw, split_urm
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.Hybrid import ScoresHybridRecommender

TRAINING_SET = "train"
NUMBER_FOLDS = 5
NUMBER_JOBS = 4
OPTIMISATION_ITERATIONS = 100

MODELS_DIR = Path("models") / TRAINING_SET
MODELS_DIR.exists()

ICM_all, URM_all, URM_train, URM_validation, URM_test = load()
if TRAINING_SET == "all":
    URM_train = URM_all

_, urm_df = load_raw()

_, urm_val_df, urm_test_df = split_urm(urm_df)
urm_valtest_df = pd.concat([urm_val_df, urm_test_df])

n_users = urm_df.user_id.nunique()
n_items = urm_df.item_id.nunique()


def urm_df_to_csr(df: pd.DataFrame) -> sps.csr_matrix:
    return sps.csr_matrix(
        (df.data, (df.user_id, df.item_id)),
        shape=(n_users, n_items),
    )


def recommender_map10(trial: optuna.Trial) -> float:
    norm = trial.suggest_categorical("norm", [1, 2, np.inf, -np.inf])
    alpha = trial.suggest_float("alpha", 0.0, 1.0)

    with (MODELS_DIR / "slim_elasticnet.pkl").open("rb") as f:
        slim_elasticnet_recommender = pickle.load(f)
    with (MODELS_DIR / "user_wide_hybrid.pkl").open("rb") as f:
        user_wide_hybrid_recommender = pickle.load(f)

    recommender_object = ScoresHybridRecommender(
        URM_train,
        slim_elasticnet_recommender,
        user_wide_hybrid_recommender,
    )
    recommender_object.fit(norm, alpha)

    fold_map = []
    for i, _ in KFold(NUMBER_FOLDS).split(urm_valtest_df):
        urm = urm_df_to_csr(urm_valtest_df.iloc[i])
        evaluator = EvaluatorHoldout(urm, cutoff_list=[10])
        result_df, _ = evaluator.evaluateRecommender(recommender_object)
        fold_map.append(result_df.loc[10]["MAP"])
    mean_map = np.mean(fold_map)
    return mean_map

study = optuna.create_study(direction="maximize")
study.optimize(
    recommender_map10,
    n_trials=OPTIMISATION_ITERATIONS,
    n_jobs=NUMBER_JOBS,
    gc_after_trial=True,
    show_progress_bar=True,
)

print(study.best_params)
print(study.best_value)
