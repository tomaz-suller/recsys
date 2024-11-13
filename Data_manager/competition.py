from pathlib import Path

import pandas as pd

from Data_manager.DataReader import DataReader
from Data_manager.DatasetMapperManager import DatasetMapperManager


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
