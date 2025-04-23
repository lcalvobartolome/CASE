

import configparser
import json
import math
import pathlib
from typing import List

import numpy as np
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
        self.fields = [col for col in  df.columns.tolist() if "researchItems" not in col]
        model_keys_to_add_to_schema = []
        for col in cols_research_items:
            
            self._logger.info(f"Processing column {col}")
            
            model = col.split("researchItems_")[1]
            ass_models = self.cf.get("aggregated-config", model).split(",")
            model_paths = [pathlib.Path(m) for m in ass_models]
            
            def get_topics_rpr(items, max_sum=1000):
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
                        
            def get_topic_rel(items):
                indices = [id_to_index[pid] for pid in items if pid in id_to_index]
                                
                if not indices:
                    return [0.0] * thetas.shape[1]

                thetas_indices = thetas[indices]
                dense = thetas_indices.toarray()

                topic_sums = dense.sum(axis=0)

                # Apply penalty
                penalty = math.log(len(indices) + 1)
                topic_rels = topic_sums / penalty

                return topic_rels.tolist()
            
            def get_tpc_rel_str(topic_rel):
                """Get the topic relevance string."""                
                rel = ""
                for idx, val in enumerate(topic_rel):
                    if val != 0:
                        rel += f"t{idx}|{val} "
                return rel.rstrip()
            
            for model_path in model_paths:
                
                try:
                    thetas = sparse.load_npz((model_path / 'TMmodel/thetas.npz'))
                    
                    tpc_rpr_key = 'agg_tpc_' + model_path.stem.lower()
                    tpc_rel_key = 'agg_rel_' + model_path.stem.lower()
                    model_keys_to_add_to_schema.append(tpc_rpr_key)
                    model_keys_to_add_to_schema.append(tpc_rel_key)
                    
                    self.models.append(model_path.stem.lower())
                    
                    # ids of publications / projects, etc.
                    with open((model_path / 'corpus.txt'), "r", encoding="utf-8") as f:
                        ids = [line.strip().split()[0] for line in f]
                    
                    id_to_index = {pid: idx for idx, pid in enumerate(ids)}
                    
                    df[tpc_rpr_key] = df[col].apply(lambda x: get_topics_rpr(x))
                    
                    self._logger.info(f"Topics for {col} calculated.")
                    self._logger.info(f"{df[['id', tpc_rpr_key]].head()}")    
                    
                    df[tpc_rel_key] = df[col].apply(lambda x: get_topic_rel(x))
                    
                    # normalize the topic relevance
                    tpc_rels = np.array(df[tpc_rel_key].tolist())
                    tpc_rels_pct = (tpc_rels / np.max(tpc_rels, axis=0))
                    
                    # save tpc_rels_pct back to df
                    df[tpc_rel_key] = tpc_rels_pct.tolist()
                    
                    # convert to str
                    df[tpc_rel_key] = df[tpc_rel_key].apply(lambda x: get_tpc_rel_str(x))
                    
                    self._logger.info(f"Topic relevance for {col} calculated.")
                    self._logger.info(f"{df[['id', tpc_rel_key]].head()}")
                    
                except Exception as e:
                    self._logger.error(f"Error processing model {model_path.stem.lower()}: {e}")
                    continue
            
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