import os
import traceback
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import (
    ItemKNN_CFCBF_Hybrid_Recommender,
)
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.RP3betaRecommenderICM import RP3betaRecommenderICM
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import (
    MatrixFactorization_BPR_Cython,
    MatrixFactorization_AsySVD_Cython,
    MatrixFactorization_SVDpp_Cython,
)
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

from Data_manager.competition import CompetitionReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Recommenders.Incremental_Training_Early_Stopping import (
    Incremental_Training_Early_Stopping,
)
from Recommenders.BaseCBFRecommender import (
    BaseItemCBFRecommender,
    BaseUserCBFRecommender,
)
from Evaluation.Evaluator import EvaluatorHoldout

OUTPUT_PATH = Path("./result_experiments/")


def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):
    if issubclass(recommender_class, BaseItemCBFRecommender) or issubclass(recommender_class, RP3betaRecommenderICM):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        # recommender_object = recommender_class(URM_train, UCM_all)
        warnings.warn(
            f"Skipping `{recommender_class}` since no user features are available"
        )
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object


def test_recommender(recommender_class, fit_params) -> str:
    log_buffer: list[str] = []

    try:
        print("Algorithm: {}".format(recommender_class))

        recommender_object = _get_instance(
            recommender_class, URM_train, ICM_all, UCM_all
        )

        if isinstance(recommender_object, Incremental_Training_Early_Stopping):
            fit_params["epochs"] = 500
            fit_params |= earlystopping_keywargs

        recommender_object.fit(**fit_params)

        results_run_1, results_run_string_1 = evaluator.evaluateRecommender(
            recommender_object
        )

        recommender_object.save_model(output_root_path, file_name="temp_model.zip")

        recommender_object = _get_instance(
            recommender_class, URM_train, ICM_all, UCM_all
        )
        recommender_object.load_model(output_root_path, file_name="temp_model.zip")

        os.remove(output_root_path + "temp_model.zip")

        results_run_2, results_run_string_2 = evaluator.evaluateRecommender(
            recommender_object
        )

        assert results_run_1.equals(results_run_2)

        print(
            "Algorithm: {}, results: \n{}".format(
                recommender_class, results_run_string_1
            )
        )
        log_buffer.append(
            "Algorithm: {}, results: \n{}\n".format(
                recommender_class, results_run_string_1
            )
        )

    except Exception as e:
        traceback.print_exc()
        log_buffer.append(
            "Algorithm: {} - Exception: {}\n".format(recommender_class, str(e))
        )

    with (OUTPUT_PATH / f"{recommender_class}.log").open("w") as f:
        f.write("\n".join(log_buffer))


if __name__ == "__main__":
    dataset_object = CompetitionReader()

    data_splitter = DataSplitter_leave_k_out(dataset_object, k_out_value=2)

    data_splitter.load_data()
    URM_train, URM_validation, URM_test = data_splitter.get_holdout_split()
    ICM_all = data_splitter.get_loaded_ICM_dict()["ICM_all"]
    UCM_all = None

    recommender_parameters = {
        SLIMElasticNetRecommender: {
            "l1_ratio": 0.4408355927953408,
            "alpha": 0.00013519978876092592,
            "positive_only": False,
            "topK": 59,
        },
        P3alphaRecommender: {
            "topK": 28,
            "alpha": 0.10129039554581817,
            "min_rating": 1,
            "implicit": True,
            "normalize_similarity": True,
        },
        RP3betaRecommenderICM: {
            "topK": 17,
            "alpha": 0.9996386762753586,
            "beta": 0.21406242351456914,
            "delta": 0.17583769441037236,
            "min_rating": 0.4181995527765843,
            "implicit": True,
            "normalize_similarity": True,
        },
        ItemKNNCFRecommender: {
            "topK": 17,
            "shrink": 144,
            "similarity": "asymmetric",
            "normalize": True,
            "asymmetric_alpha": 0.28047363677562337,
            "feature_weighting": "TF-IDF",
        },
        RP3betaRecommender: {
            "topK": 690,
            "alpha": 0.34279006187551114,
            "beta": 0.6195706675947875,
            "normalize_similarity": True,
        },
        ItemKNN_CFCBF_Hybrid_Recommender: {
            "topK": 434,
            "shrink": 5,
            "similarity": "asymmetric",
            "normalize": True,
            "asymmetric_alpha": 0.8735549428526386,
            "feature_weighting": "BM25",
            "ICM_weight": 0.020223474597100655,
        },
        SLIM_BPR_Cython: {
            "topK": 157,
            "epochs": 40,
            "symmetric": True,
            "sgd_mode": "adagrad",
            "lambda_i": 5.0090507951628164e-05,
            "lambda_j": 0.008714718021321967,
            "learning_rate": 0.03072253511293627,
        },
        UserKNNCFRecommender: {
            "topK": 565,
            "shrink": 270,
            "similarity": "cosine",
            "normalize": False,
            "feature_weighting": "BM25",
        },
        PureSVDItemRecommender: {"num_factors": 335, "topK": 957},
        MatrixFactorization_SVDpp_Cython: {
            "sgd_mode": "adagrad",
            "epochs": 500,
            "use_bias": False,
            "batch_size": 1,
            "num_factors": 183,
            "item_reg": 9.479938436299834e-05,
            "user_reg": 5.26873457199464e-05,
            "learning_rate": 0.05378457961118192,
            "negative_interactions_quota": 0.18829499083087958,
        },
        MatrixFactorization_AsySVD_Cython: {
            "sgd_mode": "sgd",
            "epochs": 20,
            "use_bias": False,
            "batch_size": 1,
            "num_factors": 116,
            "item_reg": 0.00010450121975712478,
            "user_reg": 3.4119198264653186e-05,
            "learning_rate": 0.0004915537242703813,
            "negative_interactions_quota": 0.46487536420384395,
        },
        MatrixFactorization_BPR_Cython: {
            "sgd_mode": "adagrad",
            "epochs": 10,
            "num_factors": 173,
            "batch_size": 64,
            "positive_reg": 0.00010197682788872379,
            "negative_reg": 0.00033896486580925554,
            "learning_rate": 0.002192478151949701,
        },
    }

    evaluator = EvaluatorHoldout(URM_test, [10, 20], exclude_seen=True)

    earlystopping_keywargs = {
        "validation_every_n": 5,
        "stop_on_validation": True,
        "evaluator_object": EvaluatorHoldout(URM_validation, [10], exclude_seen=True),
        "lower_validations_allowed": 5,
        "validation_metric": "MAP",
    }

    output_root_path = "./result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    logs = []
    with ProcessPoolExecutor(max_workers=8) as p:
        futures = [
            p.submit(test_recommender, recommender, parameters)
            for recommender, parameters in recommender_parameters.items()
        ]
        for future in as_completed(futures):
            try:
                logs.append(future.result())
            except Exception as e:
                traceback.print_exc()
                logs.append(str(e))
    # logs = [test_recommender(recommender) for recommender in recommender_class_list]
