import json
from pathlib import Path
import numpy as np
import datasetspecificcaveclient as dscc
from typing import Optional, Literal
import datetime
import pandas as pd

file_dir = Path(__file__)
with open(file_dir.parent / "config.json") as f:
    minnie_config = json.load(f)

ROOT_ID_LOOKUP_VIEW = "nucleus_detection_lookup_v1"
CELL_ID_LOOKUP_VIEW = "single_neurons"


class CortexClient(dscc.DatasetSpecificClient):
    def __init__(self, datastack_name, server_address):
        super().__init__(
            datastack_name=datastack_name,
            server_address=server_address,
            neuroglancer_urls={},
        )


class MinnieClient(CortexClient):
    def __init__(self):
        super().__init__(
            datastack_name=minnie_config.get("datastack_name"),
            server_address=minnie_config.get("server_address"),
        )

    def get_cell_ids(
        self,
        root_ids: list,
    ) -> np.ndarray:
        """Look up cell ids from a collection of root ids.
        Note that this will only work on neurons with a single-well defined cell body.
        Non-neuronal cells like astrocytes will not return a cell id with this function,
        but please use the soma table for the dataset to look up cell ids for them if they exist.

        Parameters
        ----------
        root_ids : list
            Collection of root ids

        Returns
        -------
        np.ndarray
            List of cell ids (or Nones) for each root id in the list

        """
        root_ids = np.array(root_ids)
        version_root_ids = self.map_root_ids_to_version(
            root_ids[root_ids != 0], version=self.version
        )
        qry = self.caveclient.materialize.views[CELL_ID_LOOKUP_VIEW]
        id_df = qry(pt_root_id=version_root_ids).query(
            select_columns=["pt_root_id", "id"]
        )
        add_ids = np.unique(
            root_ids[np.logical_not(np.isin(root_ids, id_df["pt_root_id"]))]
        )
        if len(add_ids) > 0:
            add_df = pd.DataFrame({"pt_root_id": add_ids, "id": [None] * len(add_ids)})
            id_df = pd.concat([id_df, add_df])
        return id_df.set_index("pt_root_id").loc[root_ids]["id"].values

        return id_df

    def get_root_ids(
        self,
        cell_ids: list,
        version: Optional[int] = None,
        timestamp: Optional[datetime.datetime] = None,
    ):
        if version is not None and timestamp is not None:
            raise ValueError("Cannot set both version and timestamp")
        if version is None:
            version = self.version
        if timestamp is None:
            timestamp = self.caveclient.materialize.get_timestamp(version=version)

        qry = self.caveclient.materialize.views[ROOT_ID_LOOKUP_VIEW]
        id_df = qry(id=cell_ids).query(select_columns=["pt_root_id", "id"])
        add_ids = np.unique(cell_ids[np.logical_not(np.isin(cell_ids, id_df["id"]))])
        if len(add_ids) > 0:
            add_df = pd.DataFrame({"id": add_ids, "pt_root_id": [None] * len(add_ids)})
            id_df = pd.concat([id_df, add_df])
        return id_df.set_index("id").loc[cell_ids]["pt_root_id"].values

    @property
    def spatial(self):
        pass
