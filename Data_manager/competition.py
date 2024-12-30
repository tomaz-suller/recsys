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
    icm_df, urm_df = load_raw()

    num_items = icm_df.item_id.nunique()
    num_features = icm_df.feature_id.nunique()

    icm = sp.csr_matrix(
        (icm_df.data, (icm_df.item_id, icm_df.feature_id)),
        shape=(num_items, num_features),
    )

    urm_all, urm_train, urm_validation, urm_test = split_urm_to_matrix(
        urm_df,
        validation_percentage=validation_percentage,
        testing_percentage=testing_percentage,
    )

    return icm, urm_all, urm_train, urm_validation, urm_test


def split_urm_to_matrix(
    urm_df: pd.DataFrame,
    validation_percentage: float,
    testing_percentage: float,
    seed: int = 1234,
):

    num_users = urm_df.user_id.nunique()
    num_items = urm_df.item_id.nunique()

    def df_to_csr(df: pd.DataFrame) -> sp.csr_matrix:
        return sp.csr_matrix(
            (df.data, (df.user_id, df.item_id)), shape=(num_users, num_items)
        )
    # Construct the whole URM as a sparse matrix
    urm_all = df_to_csr(urm_df)
    # Get user, item, and rating data for each set
    train_data, val_data, test_data = split_urm(
        urm_df, validation_percentage, testing_percentage, seed
    )

    # Construct sparse matrices
    urm_train = df_to_csr(train_data)
    urm_validation = df_to_csr(val_data)
    urm_test = df_to_csr(test_data)

    return urm_all, urm_train, urm_validation, urm_test


def split_urm(
    dense_urm: pd.DataFrame,
    validation_percentage: float = 0.1,
    testing_percentage: float = 0.2,
    seed: int = 1234,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split into train + validation and test sets
    train_val_indices, test_indices = train_test_split(
        np.arange(len(dense_urm)),
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

    return tuple(dense_urm.iloc[i] for i in (train_indices, val_indices, test_indices))


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    icm_df = pd.read_csv(DATA_ROOT / "data_ICM_metadata.csv")
    urm_df = pd.read_csv(DATA_ROOT / "data_train.csv")

    return icm_df, urm_df


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
