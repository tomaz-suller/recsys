from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from Data_manager.DataReader import DataReader
from Data_manager.DatasetMapperManager import DatasetMapperManager

DATA_ROOT = (
    Path(__file__).absolute().parent.parent
    / "Data_manager_split_datasets"
    / "competition"
)


def load(validation_percentage=0.10, testing_percentage=0.20):
    urm_df = pd.read_csv(DATA_ROOT / "data_train.csv")
    icm_df = pd.read_csv(DATA_ROOT / "data_ICM_metadata.csv")

    unique_users = urm_df.user_id.unique()
    num_users = unique_users.size

    unique_items = icm_df.item_id.unique()
    num_items = unique_items.size
    unique_features = icm_df.feature_id.unique()
    num_features = unique_features.size

    icm = sp.csr_matrix(
        (icm_df.data, (icm_df.item_id, icm_df.feature_id)), shape=(num_items, num_features)
    )

    urm_all, urm_train, urm_validation, urm_test = dataset_splits(
        urm_df,
        num_users=num_users,
        num_items=num_items,
        validation_percentage=validation_percentage,
        testing_percentage=testing_percentage,
    )

    return icm, urm_all, urm_train, urm_validation, urm_test


def dataset_splits(
    ratings: pd.DataFrame,
    num_users,
    num_items,
    validation_percentage: float,
    testing_percentage: float,
    seed: int = 1234,
):
    # Construct the whole URM as a sparse matrix
    urm_all = sp.csr_matrix(
        (ratings.data, (ratings.user_id, ratings.item_id)), shape=(num_users, num_items)
    )

    # Split into train + validation and test sets
    train_val_indices, test_indices = train_test_split(
        np.arange(len(ratings)),
        test_size=testing_percentage,
        shuffle=True,
        random_state=seed,
    )

    # Split train + validation into train and validation
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=validation_percentage / (1 - testing_percentage),
        shuffle=True,
        random_state=seed,
    )

    # Get user, item, and rating data for each set
    train_data = ratings.iloc[train_indices]
    val_data = ratings.iloc[val_indices]
    test_data = ratings.iloc[test_indices]

    # Construct sparse matrices
    urm_train = sp.csr_matrix(
        (train_data.data, (train_data.user_id, train_data.item_id)),
        shape=(num_users, num_items),
    )
    urm_validation = sp.csr_matrix(
        (val_data.data, (val_data.user_id, val_data.item_id)),
        shape=(num_users, num_items),
    )
    urm_test = sp.csr_matrix(
        (test_data.data, (test_data.user_id, test_data.item_id)),
        shape=(num_users, num_items),
    )

    return urm_all, urm_train, urm_validation, urm_test


class CompetitionReader(DataReader):
    DATASET_SUBFOLDER = "competition/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_all"]

    IS_IMPLICIT = True

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        dataset_path = Path(self.DATASET_SPLIT_ROOT_FOLDER) / self.DATASET_SUBFOLDER

        dataset_manager = DatasetMapperManager()

        icm_path = dataset_path / "data_ICM_metadata.csv"
        icm_df = (
            pd.read_csv(
                icm_path,
                dtype={"item_id": "string", "feature_id": "string", "data": "float32"},
            )
            .rename(
                columns={
                    "item_id": "ItemID",
                    "feature_id": "FeatureID",
                    "data": "Data",
                }
            )
            .convert_dtypes()
        )
        print("\t", "ICM information")
        print("\t", icm_df.info())
        dataset_manager.add_ICM(icm_df, "ICM_all")

        urm_path = dataset_path / "data_train.csv"
        urm_df = pd.read_csv(
            urm_path,
            dtype={"user_id": "string", "item_id": "string", "data": "float32"},
        ).rename(
            columns={
                "user_id": "UserID",
                "item_id": "ItemID",
                "data": "Data",
            }
        )
        print("\t", "URM information")
        print("\t", urm_df.info())
        dataset_manager.add_URM(urm_df, "URM_all")

        dataset = dataset_manager.generate_Dataset(
            self._get_dataset_name(), self.IS_IMPLICIT
        )
        return dataset
