import traceback
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import pairwise
from multiprocessing import cpu_count

import numpy as np
from scipy.sparse import csr_matrix

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
)
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
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


def most_interaction_user_mask(urm: csr_matrix, largest: int) -> np.ndarray:
    profile_sizes = np.ediff1d(urm.indptr)
    cutoff = np.sort(profile_sizes)[-largest]
    return profile_sizes > cutoff


def tune(args: tuple[str, csr_matrix, csr_matrix, csr_matrix]):
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    name, URM_train, URM_validation, URM_test = args
    output_folder_path = f"result_experiments_{name}/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    collaborative_algorithm_list = [
        P3alphaRecommender,
        RP3betaRecommender,
        PureSVDRecommender.PureSVDItemRecommender,
        ItemKNNCFRecommender.ItemKNNCFRecommender,
        UserKNNCFRecommender.UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender.SLIMElasticNetRecommender,
    ]

    cutoff_list = [10, 20]
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    n_cases = 100
    n_random_starts = 20

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

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

    # with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    #     futures = {
    #         executor.submit(
    #             runParameterSearch_Collaborative_partial, recommender
    #         ): recommender
    #         for recommender in collaborative_algorithm_list
    #     }
    #     for future in as_completed(futures):
    #         recommender = futures[future]
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print("On recommender {} Exception {}".format(recommender, e))
    #             traceback.print_exc()

    for recommender_class in collaborative_algorithm_list:
        try:
            runParameterSearch_Collaborative_partial(recommender_class)
        except Exception as e:
            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()

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
    NUMBER_GROUPS = 10

    dataReader = CompetitionReader()
    dataset = dataReader.load_data()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(
        dataset.get_URM_all(), train_percentage=0.80
    )
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(
        URM_train, train_percentage=0.80
    )

    number_users = URM_train.shape[0]
    group_boundary_indices: np.ndarray = np.linspace(
        number_users // 3, number_users, NUMBER_GROUPS
    ).astype("int")
    group_boundary_indices[0] = 0
    group_boundary_indices[-1] -= 1

    profile_sizes = np.ediff1d(URM_train.indptr)
    sorted_profile_sizes = np.sort(profile_sizes)

    arguments = []

    for min_, max_ in pairwise(group_boundary_indices):
        upper_bound = sorted_profile_sizes[max_]
        lower_bound = sorted_profile_sizes[min_]

        mask = np.logical_and(
            profile_sizes < upper_bound,
            profile_sizes >= lower_bound,
        )

        if (number_users := np.sum(mask)) == 0:
            continue

        URM_train_group = URM_train[mask]
        URM_validation_group = URM_validation[mask]
        URM_test_group = URM_test[mask]

        arguments.append(
            (
                f"{lower_bound}_{upper_bound}",
                URM_train_group,
                URM_validation_group,
                URM_test_group,
            )
        )

    with ProcessPoolExecutor() as pool:
        pool.map(tune, arguments)
