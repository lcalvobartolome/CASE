

import configparser
import json
import pathlib
from typing import List

from scipy import sparse
from scipy.sparse import vstack
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from src.core.entities.utils import convert_datetime_to_strftime, parseTimeINSTANT, sum_up_to

class AggregatedCorpus(object):
    """
    This class represents an aggregated corpus. Two types of AggregatedCorpus exits:
    * Researcher: ('researcher'), identifies a given researcher.
    * Research group: ('research_group'), identifies a given research group.
    """
    
    def __init__(
        self,
        path_to_raw: pathlib.Path,
        type: str,
        logger=None,
        config_file: str = "/config/config.cf"
        ) -> None:
        """Init method.
        
        Parameters
        ----------
        path_to_raw: pathlib.Path
            Path to the file with the raw corpus information of the aggregated corpus.
        type: str
            Type of the aggregated corpus. It can be 'researcher' or 'research_group'.
        logger : logging.Logger
            The logger object to log messages and errors.
        config_file: str
            Path to the configuration file.
        """
        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Entity AggregatedCorpus')
            
        if not path_to_raw.exists():
            self._logger.error(
                f"Path to raw data {path_to_raw} does not exist."
            )
            return
        
        self.path_to_raw = path_to_raw
        self.name = path_to_raw.stem.lower()
        self.type = type
        self.id_field = "invID" if type == "researcher" else "rgID"
        
        # Read configuration from config file
        self.cf = configparser.ConfigParser()
        self.cf.read(config_file)
        
        self._logger.info(f"Setting id of the aggregated corpus with name {self.name} to {self.id_field}")
    
    def get_ag_raw_info(self) -> List[dict]:
        
        ddf = dd.read_parquet(self.path_to_raw).fillna("")
        self._logger.info(ddf.head())

        ddf = ddf.rename(
            columns={
                self.id_field: "id"})
        
        with ProgressBar():
            df = ddf.compute(scheduler='processes')
            
        # prepare year columns
        df, cols = convert_datetime_to_strftime(df)
        df[cols] = df[cols].applymap(parseTimeINSTANT)
            
        # for each "researchItems{}" column... we calculate the topics
        self._logger.info("Calculating topics for each row...")
        cols_research_items = [col for col in df.columns if "researchItems" in col]
        self.models = []
        self.fields = df.columns.tolist()
        model_keys_to_add_to_schema = []
        for col in cols_research_items:
            self._logger.info(f"Processing column {col}")
            
            model = col.split("researchItems_")[1]
            model_path = pathlib.Path(self.cf.get("aggregated-config", model))
            thetas = sparse.load_npz((model_path / 'TMmodel/thetas.npz'))
            
            model_key = 'agg_tpc_' + model.lower()
            model_keys_to_add_to_schema.append(model_key)
            
            self.models.append(model.lower())
            
            # ids of publications / projects, etc.
            with open((model_path / 'corpus.txt'), "r", encoding="utf-8") as f:
                ids = [line.strip().split()[0] for line in f]
            
            id_to_index = {pid: idx for idx, pid in enumerate(ids)}
            
            def get_topics(items, max_sum=1000):
                """Get the topics for a given list of items."""
                # items is the list of research items associated with the row
                indices = [id_to_index[pid] for pid in items if pid in id_to_index]
                
                mean_vector = None
                if indices:
                    subset_matrix = vstack([thetas[i] for i in indices])
                    mean_vector = subset_matrix.mean(axis=0)  # 1 x n_topics
                    mean_vector = mean_vector.A1  # Convert to 1D array
                    mean_vector = sum_up_to(mean_vector, max_sum)
                    
                    # Convert to string representation
                    rpr = ""
                    for idx, val in enumerate(mean_vector):
                        if val != 0:
                            rpr += "t" + str(idx) + "|" + str(val) + " "
                    rpr = rpr.rstrip()
                    
                return rpr if mean_vector is not None else ""
            
            df[model_key] = df[col].apply(lambda x: get_topics(x))
            
            self._logger.info(f"Topics for {col} calculated.")
            self._logger.info(f"{df[['id', model_key]].head()}")
            
        json_str = df.to_json(orient='records')
        json_lst = json.loads(json_str)
            
        return json_lst, model_keys_to_add_to_schema
            
    def get_agg_corpora_update(self, id:int) -> List[dict]:
        """Get the update for the aggregated corpus."""
        
        fields_dict = [{"id": id,
                        "type": self.type,
                        "agg_name": self.name,
                        "agg_path": self.path_to_raw.as_posix(),
                        "models": self.models,
                        "fields": self.fields
                        }]
        
        return fields_dict