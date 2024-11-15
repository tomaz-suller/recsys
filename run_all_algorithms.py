import os
import traceback
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import (
    UserKNN_CFCBF_Hybrid_Recommender,
)
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import (
    ItemKNN_CFCBF_Hybrid_Recommender,
)
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

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
    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        # recommender_object = recommender_class(URM_train, UCM_all)
        warnings.warn(
            f"Skipping `{recommender_class}` since no user features are available"
        )
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object


def test_recommender(recommender_class) -> str:
    log_buffer: list[str] = []

    try:
        print("Algorithm: {}".format(recommender_class))

        recommender_object = _get_instance(
            recommender_class, URM_train, ICM_all, UCM_all
        )

        if isinstance(recommender_object, Incremental_Training_Early_Stopping):
            fit_params = {"epochs": 15, **earlystopping_keywargs}
        else:
            fit_params = {}

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

    dataSplitter = DataSplitter_leave_k_out(dataset_object, k_out_value=2)

    dataSplitter.load_data()
    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    ICM_all = dataSplitter.get_loaded_ICM_dict()["ICM_all"]
    UCM_all = None

    recommender_class_list = [
        UserKNNCBFRecommender,
        ItemKNNCBFRecommender,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        UserKNN_CFCBF_Hybrid_Recommender,
        ItemKNN_CFCBF_Hybrid_Recommender,
        SLIMElasticNetRecommender,
        PureSVDRecommender,
        NMFRecommender,
        IALSRecommender,
        # EASE_R_Recommender,
        # P3alphaRecommender,
        # RP3betaRecommender,
    ]

    evaluator = EvaluatorHoldout(URM_test, [5, 20], exclude_seen=True)

    earlystopping_keywargs = {
        "validation_every_n": 5,
        "stop_on_validation": True,
        "evaluator_object": EvaluatorHoldout(URM_validation, [20], exclude_seen=True),
        "lower_validations_allowed": 5,
        "validation_metric": "MAP",
    }

    output_root_path = "./result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    logs = []
    with ProcessPoolExecutor() as p:
        futures = [
            p.submit(test_recommender, recommender)
            for recommender in recommender_class_list
        ]
        for future in as_completed(futures):
            try:
                logs.append(future.result())
            except Exception as e:
                traceback.print_exc()
                logs.append(str(e))
    # logs = [test_recommender(recommender) for recommender in recommender_class_list]
