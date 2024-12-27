import traceback
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np

from Recommenders.NonPersonalizedRecommender import Random, TopPop
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.KNN import (
    UserKNNCFRecommender,
    ItemKNNCFRecommender,
    ItemKNNCBFRecommender,
)
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import (
    ItemKNN_CFCBF_Hybrid_Recommender,
)
from Recommenders.MatrixFactorization import (
    PureSVDRecommender,
)
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import (
    MatrixFactorization_BPR_Cython,
    MatrixFactorization_AsySVD_Cython,
    MatrixFactorization_SVDpp_Cython,
    MatrixFactorization_WARP_Cython,
)
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.RP3betaRecommenderICM import RP3betaRecommenderICM
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.run_hyperparameter_search import (
    runHyperparameterSearch_Collaborative,
    runHyperparameterSearch_Content,
    runHyperparameterSearch_Hybrid,
)
from Data_manager.competition import load


def read_data_split_and_search(users_to_ignore: list[int], output_path: Path):
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """
    icm, _, urm_train, urm_val, _ = load()

    ICM_name = "ICM_all"
    ICM_object = icm
    URM_train = urm_train
    URM_validation = urm_val
    output_folder_path = str(output_path)

    collaborative_algorithm_list = [
        Random,
        TopPop,
        PureSVDRecommender.PureSVDItemRecommender,
        ItemKNNCFRecommender.ItemKNNCFRecommender,
        UserKNNCFRecommender.UserKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        SLIMElasticNetRecommender.SLIMElasticNetRecommender,
        SLIM_BPR_Cython,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_SVDpp_Cython,
        # MatrixFactorization_WARP_Cython,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_AsySVD_Cython
    ]

    hybrid_algorithm_list = [
        # RP3betaRecommenderICM,
        # ItemKNN_CFCBF_Hybrid_Recommender,
    ]

    cutoff_list = [10, 20, 30]
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    n_cases = 100
    n_random_starts = 20

    evaluator_validation = EvaluatorHoldout(
        URM_validation,
        cutoff_list=cutoff_list,
        ignore_users=users_to_ignore,
    )

    runParameterSearch_Collaborative_partial = partial(
        runHyperparameterSearch_Collaborative,
        URM_train=URM_train,
        metric_to_optimize=metric_to_optimize,
        cutoff_to_optimize=cutoff_to_optimize,
        n_cases=n_cases,
        n_random_starts=n_random_starts,
        evaluator_validation_earlystopping=evaluator_validation,
        evaluator_validation=evaluator_validation,
        evaluator_test=None,
        output_folder_path=output_folder_path,
        resume_from_saved=True,
        similarity_type_list=None,
        parallelizeKNN=True,
        evaluate_on_test="no",
    )

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {
            executor.submit(
                runParameterSearch_Collaborative_partial, recommender
            ): recommender
            for recommender in collaborative_algorithm_list
        }
        for future in as_completed(futures):
            recommender = futures[future]
            try:
                future.result()
            except Exception as e:
                print("On recommender {} Exception {}".format(recommender, e))
                traceback.print_exc()

    # for recommender_class in collaborative_algorithm_list:
    #     try:
    #         runParameterSearch_Collaborative_partial(recommender_class)
    #     except Exception as e:
    #         print("On recommender {} Exception {}".format(recommender_class, str(e)))
    #         traceback.print_exc()

    ################################################################################################
    ###### Content Baselines

    for ICM_name, ICM_object in dataset.get_loaded_ICM_dict().items():
        try:
            runHyperparameterSearch_Content(
                ItemKNNCBFRecommender,
                URM_train=URM_train,
                URM_train_last_test=URM_train + URM_validation,
                metric_to_optimize=metric_to_optimize,
                cutoff_to_optimize=cutoff_to_optimize,
                evaluator_validation=evaluator_validation,
                evaluator_test=evaluator_test,
                output_folder_path=output_folder_path,
                parallelizeKNN=True,
                allow_weighting=True,
                resume_from_saved=True,
                similarity_type_list=["cosine"],
                ICM_name=ICM_name,
                ICM_object=ICM_object.copy(),
                n_cases=n_cases,
                n_random_starts=n_random_starts,
            )

        except Exception as e:
            print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
            traceback.print_exc()

        try:
            runHyperparameterSearch_Hybrid(
                ItemKNN_CFCBF_Hybrid_Recommender,
                URM_train=URM_train,
                URM_train_last_test=URM_train + URM_validation,
                metric_to_optimize=metric_to_optimize,
                cutoff_to_optimize=cutoff_to_optimize,
                evaluator_validation=evaluator_validation,
        evaluator_test=None,
        output_folder_path=output_folder_path,
        parallelizeKNN=True,
        allow_weighting=True,
        resume_from_saved=True,
        similarity_type_list=None,
        ICM_name=ICM_name,
        ICM_object=ICM_object.copy(),
        n_cases=n_cases,
        n_random_starts=n_random_starts,
        evaluate_on_test="no",
    )

    for recommender_class in hybrid_algorithm_list:
        try:
            runParameterSearch_Hybrid_partial(recommender_class)
        except Exception as e:
            print(
                "On recommender {} Exception {}".format(
                    recommender_class, str(e)
                )
            )
            traceback.print_exc()


if __name__ == "__main__":
    NUMBER_BLOCKS = 10

    base_output_path = Path("results") / datetime.now().isoformat()
    base_output_path.mkdir(parents=True, exist_ok=False)

    _, _, urm_train, _, _ = load()
    profile_lengths = np.ediff1d(urm_train.indptr)
    sorted_users = np.argsort(profile_lengths)
    block_size = len(sorted_users) // NUMBER_BLOCKS
    for i in range(NUMBER_BLOCKS+1):
        users_to_consider = sorted_users[i * block_size : (i + 1) * block_size]
        if users_to_consider.size == 0:
            break
        users_to_ignore = np.setdiff1d(sorted_users, users_to_consider)
        output_path = base_output_path / f"block_{i}"
        output_path.mkdir()
        read_data_split_and_search(users_to_ignore, output_path)
