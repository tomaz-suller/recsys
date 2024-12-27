import numpy.linalg as LA
import scipy.sparse as sps

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import (
    ItemKNNCustomSimilarityRecommender,
)

# recommender_object = ItemKNNCustomSimilarityRecommender(URM_train)
# recommender_object.fit(new_similarity)

# result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
# result_df


class ScoresHybridRecommender(BaseRecommender):
    """
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2
    may come from algorithms trained on different loss functions.
    In this case, normalising scores is advisable (by default, no normalisation is applied.)
    """

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super().__init__(URM_train)
        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.norm = None

    def fit(self, norm=None, alpha=0.5):
        self.alpha = alpha
        self.norm = norm

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        if self.norm is not None:
            norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
            if norm_item_weights_1 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(
                        self.norm
                    )
                )
            item_weights_1 /= norm_item_weights_1

            norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
            if norm_item_weights_2 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(
                        self.norm
                    )
                )
            item_weights_2 /= norm_item_weights_2

        item_weights = (
            item_weights_1  * self.alpha
            + item_weights_2 * (1 - self.alpha)
        )

        return item_weights


class UserWideHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "UserWideHybridRecommender"

    def __init__(self, URM_train, group_users: dict, group_recommenders: dict[int, BaseRecommender], verbose=True):
        super().__init__(URM_train, verbose=verbose)
        self.user_groups = {user: group for group, users in group_users.items() for user in users}
        self.group_recommenders = group_recommenders

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        for user_id in user_id_array:
            user_group = self.user_groups[user_id]
            return self.group_recommenders[user_group]._compute_item_score(user_id_array, items_to_compute)


# # %% [markdown] slideshow={"slide_type": "slide"}
# # # Some tricks, user-wise hybrids
# #
# # Models do not have the same accuracy for different user types. Let's divide the users according to their profile length and then compare the recommendation quality we get from a CF model.
# #
# # Let's categorize user based on the number of interactions they have.

# # %%
# import numpy as np
# import scipy.sparse as sps

# profile_length = np.ediff1d(sps.csr_matrix(URM_train).indptr)
# profile_length, profile_length.shape

# # %% [markdown] slideshow={"slide_type": "subslide"}
# # Let's select a few groups of 5% of the users with the least number of interactions.

# # %%
# block_size = int(len(profile_length) * 0.05)
# block_size

# # %%
# sorted_users = np.argsort(profile_length)
# sorted_users

# # %% slideshow={"slide_type": "fragment"}
# for group_id in range(0, 20):
#     start_pos = group_id * block_size
#     end_pos = min((group_id + 1) * block_size, len(profile_length))

#     users_in_group = sorted_users[start_pos:end_pos]

#     users_in_group_p_len = profile_length[users_in_group]

#     print(
#         "Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
#             group_id,
#             users_in_group.shape[0],
#             users_in_group_p_len.mean(),
#             np.median(users_in_group_p_len),
#             users_in_group_p_len.min(),
#             users_in_group_p_len.max(),
#         )
#     )

# # %% [markdown]
# # Now let's calculate the evaluation metrics of each recommender when considering groups of users.

# # %%
# from Recommenders.NonPersonalizedRecommender import TopPop
# from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
# from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
# from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
# from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
# from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
# from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import (
#     MatrixFactorization_FunkSVD_Cython,
# )
# from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
# from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender

# MAP_recommender_per_group = {}

# collaborative_recommender_class = {
#     "TopPop": TopPop,
#     "UserKNNCF": UserKNNCFRecommender,
#     "ItemKNNCF": ItemKNNCFRecommender,
#     "P3alpha": P3alphaRecommender,
#     "RP3beta": RP3betaRecommender,
#     "PureSVD": PureSVDRecommender,
#     "NMF": NMFRecommender,
#     "FunkSVD": MatrixFactorization_FunkSVD_Cython,
#     "SLIMBPR": SLIM_BPR_Cython,
# }

# content_recommender_class = {
#     "ItemKNNCBF": ItemKNNCBFRecommender,
#     "ItemKNNCFCBF": ItemKNN_CFCBF_Hybrid_Recommender,
# }

# recommender_object_dict = {}

# for label, recommender_class in collaborative_recommender_class.items():
#     recommender_object = recommender_class(URM_train)
#     recommender_object.fit()
#     recommender_object_dict[label] = recommender_object

# for label, recommender_class in content_recommender_class.items():
#     recommender_object = recommender_class(URM_train, ICM_genres)
#     recommender_object.fit()
#     recommender_object_dict[label] = recommender_object

# # %%
# cutoff = 10

# for group_id in range(0, 20):
#     start_pos = group_id * block_size
#     end_pos = min((group_id + 1) * block_size, len(profile_length))

#     users_in_group = sorted_users[start_pos:end_pos]

#     users_in_group_p_len = profile_length[users_in_group]

#     print(
#         "Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
#             group_id,
#             users_in_group.shape[0],
#             users_in_group_p_len.mean(),
#             np.median(users_in_group_p_len),
#             users_in_group_p_len.min(),
#             users_in_group_p_len.max(),
#         )
#     )

#     users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
#     users_not_in_group = sorted_users[users_not_in_group_flag]

#     evaluator_test = EvaluatorHoldout(
#         URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group
#     )

#     for label, recommender in recommender_object_dict.items():
#         result_df, _ = evaluator_test.evaluateRecommender(recommender)
#         if label in MAP_recommender_per_group:
#             MAP_recommender_per_group[label].append(result_df.loc[cutoff]["MAP"])
#         else:
#             MAP_recommender_per_group[label] = [result_df.loc[cutoff]["MAP"]]

# # %%
# import matplotlib.pyplot as plt

# # %matplotlib inline

# _ = plt.figure(figsize=(16, 9))
# for label, recommender in recommender_object_dict.items():
#     results = MAP_recommender_per_group[label]
#     plt.scatter(x=np.arange(0, len(results)), y=results, label=label)
# plt.ylabel("MAP")
# plt.xlabel("User Group")
# plt.legend()
# plt.show()

# # %% [markdown]
# # ### The recommendation quality of the three algorithms changes depending on the user profile length
# #
# # ## Tip:
# # ### If an algorithm works best on average, it does not imply it will work best for ALL user types
