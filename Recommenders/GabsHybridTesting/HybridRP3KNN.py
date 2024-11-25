#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder
import time
import sys

class HybridRP3betaKNNRecommender(BaseItemSimilarityMatrixRecommender):
    """ 
    Hybrid recommender that combines RP3betaICM and ItemKNNCF
    through a weighted sum of their similarity matrices
    """
    
    RECOMMENDER_NAME = "HybridRP3betaKNNRecommender"

    def __init__(self, URM_train, rp3beta_icm_recommender, knn_recommender, verbose=True):
        """
        Initialize the recommender
        
        Args:
            URM_train: user-rating matrix
            rp3beta_icm_recommender: trained RP3betaICM recommender
            knn_recommender: trained ItemKNNCF recommender
        """
        super(HybridRP3betaKNNRecommender, self).__init__(URM_train, verbose=verbose)
        
        self.rp3beta_icm_recommender = rp3beta_icm_recommender
        self.knn_recommender = knn_recommender
        
        if self.verbose:
            print(f"{self.RECOMMENDER_NAME}: Initialized")

    def fit(self, alpha=0.5, normalize_similarity=True):
        """
        Combine the two recommenders
        
        Args:
            alpha: float between 0 and 1
                Weight for RP3betaICM recommender
                (1-alpha) will be the weight for ItemKNNCF
            normalize_similarity: bool
                Whether to normalize the final similarity matrix
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")

        self.alpha = alpha
        
        if self.verbose:
            print(f"{self.RECOMMENDER_NAME}: Combining matrices with alpha={alpha}")

        # Combine the similarity matrices
        self.W_sparse = alpha * self.rp3beta_icm_recommender.W_sparse + \
                       (1-alpha) * self.knn_recommender.W_sparse

        # Optionally normalize the combined similarity matrix
        if normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')
        
        if self.verbose:
            print(f"{self.RECOMMENDER_NAME}: Computation completed")