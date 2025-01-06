import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import tqdm

from Data_manager.competition import load, load_raw
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Hybrid import (
    ScoresMultipleHybridRecommender,
    UserWideHybridRecommender,
)
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity

USE = "training"
OUTPUT_PATH = Path() / f"ranker_{USE}_data_test.parquet"
NUMBER_FOLDS = None
CUTOFF = 10

ITEM_LATENT_DIMENSIONS = 5
USER_LATENT_DIMENSIONS = 5

MODELS_BASE_DIR = Path() / "models"
TRAIN_MODELS_DIR = MODELS_BASE_DIR / "train" / "map"
SUBMISSION_MODELS_DIR = MODELS_BASE_DIR / "all" / "map" / "0"

USER_WIDE_HYBRID_BEGIN = 30
NUMBER_GROUPS_USER_WIDE_HYBRID = 10

MODELS_TO_USE = None
# MODELS_TO_USE = (
#     60,
#     61,
#     62,
#     63,
# )

MULTIPLE_SCORE_HYBRID_WEIGHTS = {
    50: 0.253770701546336,
    51: 0.10324855050317669,
}

SVD_FIT_PARAMS = {
    "num_factors": 350,
}


def build_user_wide_hybrid(urm: sps.csr_matrix, models: dict[str, BaseRecommender]):
    profile_lengths = np.ediff1d(urm.indptr)
    sorted_users = np.argsort(profile_lengths)
    block_size = len(sorted_users) // NUMBER_GROUPS_USER_WIDE_HYBRID
    group_users = {}
    for group in range(NUMBER_GROUPS_USER_WIDE_HYBRID + 1):
        group_users[group] = sorted_users[group * block_size : (group + 1) * block_size]
    group_recommenders = {
        group: models.pop(str(USER_WIDE_HYBRID_BEGIN + group))
        for group in range(NUMBER_GROUPS_USER_WIDE_HYBRID + 1)
    }
    return UserWideHybridRecommender(urm, group_users, group_recommenders)


def build_score_hybrid(urm: sps.csr_matrix, models: dict[str, BaseRecommender]):
    recommenders = [
        models.pop(str(index)) for index in MULTIPLE_SCORE_HYBRID_WEIGHTS.keys()
    ]
    weights = list(MULTIPLE_SCORE_HYBRID_WEIGHTS.values())
    return ScoresMultipleHybridRecommender(urm, recommenders, weights)


def urm_df_to_csr(
    urm_df: pd.DataFrame, number_users: int, number_items: int
) -> sps.csr_matrix:
    return sps.csr_matrix(
        (urm_df.data, (urm_df.user_id, urm_df.item_id)),
        shape=(number_users, number_items),
    )


def compute_base_dataset(
    number_users: int,
    models: dict[str, BaseRecommender],
    cutoff: int,
    fold: Optional[int] = None,
) -> pd.DataFrame:
    dataset = pd.DataFrame(index=range(0, number_users), columns=["ItemID"])
    dataset.index.name = "UserID"

    recommendations_list = []
    recommenders_list = []
    rank_list = []
    for user_id in tqdm(range(number_users), desc="User (candidate)"):
        user_recommendations = []
        user_recommenders = []
        user_rankings = []
        for name, recommender in models.items():
            user_recommendations.extend(
                recommender.recommend(
                    user_id,
                    cutoff=cutoff,
                    remove_seen_flag=True,
                )
            )
            user_recommenders.extend([name] * cutoff)
            user_rankings.extend(list(range(cutoff)))
        recommendations_list.append(user_recommendations)
        recommenders_list.append(user_recommenders)
        rank_list.append(user_rankings)

    dataset["ItemID"] = recommendations_list
    dataset["Recommender"] = recommenders_list
    dataset["Ranking"] = rank_list

    exploded_recommender = dataset["Recommender"].explode()
    exploded_ranking = dataset["Ranking"].explode()
    dataset = dataset.explode("ItemID")
    dataset["Recommender"] = exploded_recommender
    dataset["Ranking"] = exploded_ranking.astype("int")

    recommender_agreement = (
        dataset.reset_index()[["UserID", "ItemID"]]
        .groupby(["UserID", "ItemID"])
        .value_counts()
    )
    dataset["recommender_agreement"] = recommender_agreement.loc[
        list(zip(dataset.index, dataset["ItemID"]))
    ].to_numpy()

    for user_id in tqdm(dataset.index.unique(), desc="User (score)"):
        for rec_label, rec_instance in models.items():
            item_list = dataset.loc[user_id, "ItemID"].to_list()

            all_item_scores = rec_instance._compute_item_score(
                [user_id], items_to_compute=item_list
            )

            dataset.loc[user_id, rec_label] = all_item_scores[0, item_list]

    dataset = dataset.reset_index()
    dataset = dataset.rename(columns={"index": "UserID"})

    if fold is not None:
        dataset["fold"] = fold

    return dataset


def add_labels(training_df: pd.DataFrame, correct_recommendations_df: pd.DataFrame):
    training_df = training_df.merge(
        correct_recommendations_df,
        on=["UserID", "ItemID"],
        how="left",
        indicator="Exist",
    )
    training_df["Label"] = training_df["Exist"] == "both"
    training_df = training_df.drop(columns=["Exist"])

    return training_df


def compute_correct_recommendations(
    urm_val: sps.csr_matrix,
) -> pd.DataFrame:
    urm_val_coo = sps.coo_matrix(urm_val)
    return pd.DataFrame({"UserID": urm_val_coo.row, "ItemID": urm_val_coo.col})


def load_models_fold(
    urm: sps.csr_matrix,
    fold_dir: Path,
    use_only: Optional[list[str]] = None,
    with_user_hybrid: bool = False,
    with_score_hybrid: bool = False,
):
    all_models = load_models_all(fold_dir)
    models = all_models
    if use_only is not None:
        models = {str(index): models[str(index)] for index in use_only}
    if with_user_hybrid and "user_wide_hybrid" not in models:
        user_wide_hybrid = build_user_wide_hybrid(urm, all_models)
        models["user_wide_hybrid"] = user_wide_hybrid
    if with_score_hybrid and "score_hybrid" not in models:
        score_hybrid = build_score_hybrid(urm, all_models)
        models["score_hybrid"] = score_hybrid

    return models


def load_models_all(dir_: Path) -> dict[str, BaseRecommender]:
    return {path.stem: pickle.load(path.open("rb")) for path in dir_.glob("*.pkl")}


def compute_training_dataset(
    number_users: int,
    number_items: int,
    urm_df: pd.DataFrame,
    folds: Optional[int] = None,
):
    if folds is None:
        split = ShuffleSplit(1, test_size=0.2, random_state=42)
    else:
        split = KFold(folds, shuffle=True, random_state=42)

    fold_training_datasets: dict[int, pd.DataFrame] = {}
    for i, (train_indices, val_indices) in tqdm(
        enumerate(split.split(urm_df)),
        total=folds,
        desc="Fold",
    ):
        fold_urm_train = urm_df_to_csr(
            urm_df.iloc[train_indices], number_users, number_items
        )
        fold_urm_val = urm_df_to_csr(
            urm_df.iloc[val_indices], number_users, number_items
        )

        fold_models_dir = TRAIN_MODELS_DIR
        if folds is not None:
            fold_models_dir /= str(i)

        models = load_models_fold(
            fold_urm_train,
            fold_models_dir,
            use_only=MODELS_TO_USE,
            with_user_hybrid=True,
            with_score_hybrid=False,
        )

        fold_training_dataset = compute_base_dataset(
            number_users,
            models,
            CUTOFF,
            fold=i,
        )
        correct_recommendations_df = compute_correct_recommendations(fold_urm_val)
        fold_training_dataset = add_labels(
            fold_training_dataset, correct_recommendations_df
        )
        fold_training_datasets[i] = fold_training_dataset

    return pd.concat(fold_training_datasets.values())


def compute_submission_dataset(number_users: int):
    models = load_models_all(SUBMISSION_MODELS_DIR)
    return compute_base_dataset(
        number_users,
        models,
        CUTOFF,
    )


def compute_dataset(
    use: Literal["training", "submission"],
    number_users: int,
    number_items: int,
    urm_df: pd.DataFrame,
    folds: Optional[int] = None,
):
    if use == "training":
        return compute_training_dataset(number_users, number_items, urm_df, folds)
    elif use == "submission":
        return compute_submission_dataset(number_users)


def add_features(dataset: pd.DataFrame, urm: sps.csr_matrix, icm: sps.csr_matrix):
    svd = PureSVDRecommender(urm)
    svd.fit(**SVD_FIT_PARAMS)
    # Item features

    ## Item popularity
    item_popularity = np.ediff1d(sps.csc_matrix(urm).indptr)

    dataset["item_popularity"] = item_popularity[
        dataset["ItemID"].to_numpy().astype(int)
    ]

    ## Distance to closest items

    item_similarity = Compute_Similarity(icm.T).compute_similarity()
    item_similarity

    mean_item_similarity_dict = {i: row.mean() for i, row in enumerate(item_similarity)}
    mean_item_similarity: pd.DataFrame = pd.Series(mean_item_similarity_dict).to_frame(
        name="item_similarity"
    )
    mean_item_similarity

    dataset = dataset.join(mean_item_similarity, on="ItemID")

    ## Singular vectors
    for i in range(ITEM_LATENT_DIMENSIONS):
        dataset[f"item_svd_{i}"] = svd.ITEM_factors[
            dataset["ItemID"].to_numpy().astype(int), i
        ]

    # User features

    ## User popularity

    user_popularity = np.ediff1d(sps.csr_matrix(urm).indptr)

    dataset["user_profile_len"] = user_popularity[
        dataset["UserID"].to_numpy().astype(int)
    ]

    ## User popularity bias
    # (measure of how much popularity influences the user)

    item_popularity_ranking = item_popularity.argsort()[::-1]
    item_popularity_ranking

    item_id_df = urm_df[["user_id", "item_id"]]
    item_id_df

    TOP_POPULAR_THRESHOLDS = (10, 100, 1000)

    for k in TOP_POPULAR_THRESHOLDS:
        top_k_popular = item_popularity_ranking[:k]
        item_id_df.loc[item_id_df["item_id"].isin(top_k_popular), f"top_{k}"] = 1
    item_id_df = item_id_df.fillna(0)
    item_id_df

    user_top_k_df = item_id_df.groupby("user_id").aggregate(
        {f"top_{k}": "sum" for k in TOP_POPULAR_THRESHOLDS}
    )
    user_top_k_df

    dataset = dataset.join(user_top_k_df, on="UserID")

    ## Distance to closest users

    user_similarity = Compute_Similarity(urm.T).compute_similarity()
    user_similarity

    mean_user_similarity_dict = {i: row.mean() for i, row in enumerate(user_similarity)}
    mean_user_similarity: pd.DataFrame = pd.Series(mean_user_similarity_dict).to_frame(
        name="user_similarity"
    )
    mean_user_similarity

    dataset = dataset.join(mean_user_similarity, on="UserID")

    ## Singular vectors
    for i in range(USER_LATENT_DIMENSIONS):
        dataset[f"user_svd_{i}"] = svd.USER_factors[
            dataset["UserID"].to_numpy().astype(int), i
        ]

    return dataset


if __name__ == "__main__":
    icm_df, urm_df = load_raw()
    number_users = urm_df["user_id"].nunique()
    number_items = icm_df["item_id"].nunique()

    icm_matrix, urm_all, *_ = load()

    dataset = compute_dataset(
        use=USE,
        number_users=number_users,
        number_items=number_items,
        urm_df=urm_df,
        folds=NUMBER_FOLDS,
    )

    dataset = add_features(dataset, urm_all, icm_matrix)

    for categorical_column in ("UserID", "ItemID", "Recommender"):
        dataset[categorical_column] = dataset[categorical_column].astype("category")

    dataset.to_parquet(OUTPUT_PATH)
