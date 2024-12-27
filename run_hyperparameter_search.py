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
from Data_manager.competition import CompetitionReader
from Data_manager.split_functions.split_train_validation_random_holdout import (
    split_train_in_two_percentage_global_sample,
)


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    dataReader = CompetitionReader()
    dataset = dataReader.load_data()

    # TODO Consider removing test to train with more data
    URM_train, URM_test = split_train_in_two_percentage_global_sample(
        dataset.get_URM_all(), train_percentage=0.80
    )
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(
        URM_train, train_percentage=0.80
    )

    profile_sizes = np.ediff1d(URM_train.indptr)
    sorted_profile_indices = np.argsort(profile_sizes)
    big_profile_users = sorted_profile_indices[URM_train.shape[0]//2:]

    output_folder_path = Path("results") / datetime.now().isoformat()
    output_folder_path.mkdir(parents=True, exist_ok=False)
    # For compatibility with the rest of the framework
    output_folder_path = str(output_folder_path)

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

    cutoff_list = [10, 20]
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    n_cases = 100
    n_random_starts = 20

    evaluator_validation = EvaluatorHoldout(
        URM_validation,
        cutoff_list=cutoff_list,
        ignore_users=big_profile_users,
    )
    evaluator_test = EvaluatorHoldout(
        URM_test,
        cutoff_list=cutoff_list,
        ignore_users=big_profile_users,
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
        evaluator_test=evaluator_test,
        output_folder_path=output_folder_path,
        resume_from_saved=True,
        similarity_type_list=None,
        parallelizeKNN=False,
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
                evaluator_test=evaluator_test,
                output_folder_path=output_folder_path,
                parallelizeKNN=True,
                allow_weighting=True,
                resume_from_saved=True,
                similarity_type_list=None,
                ICM_name=ICM_name,
                ICM_object=ICM_object.copy(),
                n_cases=n_cases,
                n_random_starts=n_random_starts,
            )

        except Exception as e:
            print(
                "On recommender {} Exception {}".format(
                    ItemKNN_CFCBF_Hybrid_Recommender, str(e)
                )
            )
            traceback.print_exc()


if __name__ == "__main__":
    read_data_split_and_search()
