"""
This module provides the class Inferencer, which consists of a wrapper for perfoming inference on a new unseen corpus. It contains specific implementations according to the trainer used for the generation of the topic model that is being used for inference.

Author: Lorena Calvo-Bartolomé
Date: 19/05/2023
"""

import argparse
import json
import logging
import os
import pathlib
import sys
from abc import abstractmethod
from pathlib import Path
from subprocess import check_output
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from src.core.models.neural_models.contextualized_topic_models.utils.data_preparation import \
    prepare_hold_out_dataset
from src.core.models.neural_models.pytorchavitm.utils.data_preparation import \
    prepare_hold_out_dataset
from src.core.utils import sum_up_to, unpickler_avitm_for_case_inferencer


class Inferencer(object):
    """
    Wrapper for a Generic Topic Model Inferencer
    Assumes input is given by a inferConfigFile with the following format:
    infer_config = {
        "description": Description of the inference,
        "infer_path": Path where the inference will be saved,
        "model_for_infer_path": Path to the model to be used for inference,
        "trainer": Model trainer that will be used for inference,
        "TrDtSet": Path to the dataset used for training,
        "text_to_infer": Text to infer,
        "Preproc": Preprocessing object,
        "TMparam": Parameters of the topic model,
        "creation_date": Creation date of the inference,
        }
    """

    def __init__(self,
                 logger: logging.Logger) -> None:
        """
        Initilization Method

        Parameters
        ----------
        logger: Logger object
            To log object activity
        """

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Inferencer')

        return

    def transform_inference_output(self,
                                   thetas32: np.array,
                                   max_sum: int) -> List[dict]:
        """Saves the topic distribution for each document in text format (tXX|weightXX)

        Parameters
        ----------
        thetas32: np.ndarray
            Doc-topic distribution of the inferred documents

        Returns
        -------
        List[dict]: List of dictionaries with the topic distribution for each document
        """

        self._logger.info(
            '-- Inference: Saving the topic distribution in text format')

        def get_doc_str_rpr(vector: np.array, max_sum: int) -> str:
            """Calculates the string representation of a document's topic proportions in the format 't0|100 t1|200 ...', so that the sum of the topic proportions is at most max_sum.

            Parameters
            ----------
            vector: numpy.array
                Array with the topic proportions of a document.
            max_sum: int
                Maximum sum of the topic proportions.

            Returns 
            -------
            rpr: str
                String representation of the document's topic proportions.
            """
            vector = sum_up_to(vector, max_sum)
            rpr = ""
            for idx, val in enumerate(vector):
                if val != 0:
                    rpr += "t" + str(idx) + "|" + str(val) + " "
            rpr = rpr.rstrip()
            return rpr

        if thetas32.ndim == 2:
            doc_tpc_rpr = [get_doc_str_rpr(thetas32[row, :], max_sum)
                           for row in range(len(thetas32))]
        elif thetas32.ndim == 1:
            doc_tpc_rpr = [get_doc_str_rpr(thetas32, max_sum)]
        else:
            self._logger.error(
                f"-- -- Thetas32 has wrong number of dimensions when transforming inference output")
        ids = np.arange(len(thetas32))
        df = pd.DataFrame(list(zip(ids, doc_tpc_rpr)),
                          columns=['id', 'thetas'])

        infer_path = Path(self._inferConfig["infer_path"])
        doc_topics_file_csv = infer_path.joinpath("doc-topics.csv")
        df.to_csv(doc_topics_file_csv, index=False)

        return df.to_dict(orient='records')

    def apply_model_editions(self, thetas32: np.ndarray) -> None:
        """Load thetas file, apply model edition actions, and save it as a numpy array

        Parameters
        ----------
        thetas32: np.ndarray
            Doc-topic distribution of the inferred documents
        """
        self._logger.info(
            f'-- Inference: Applying model edition transformations')

        model_for_infer_path = Path(self._inferConfig["model_for_infer_path"])
        infer_path = Path(self._inferConfig["infer_path"])

        model_edits = model_for_infer_path.joinpath('TMmodel/edits.txt')
        self._logger.info(
            f'-- Model edits: {model_edits.as_posix()}')
        if model_edits.is_file():
            with model_edits.open('r', encoding='utf8') as fin:
                for line in fin:
                    self._logger.info(f'--Line: {line}')
                    line_els = line.strip().split()
                    if line_els[0] == 's':
                        idx = [int(el) for el in line_els[1:]]
                        if thetas32.ndim == 2:
                            self._logger.info(f'-- Thetas dim 2 before')
                            self._logger.info(f'-- Thetas32: {thetas32}')
                            thetas32 = thetas32[idx, :]
                            self._logger.info(f'-- Thetas dim 2 after')
                        elif thetas32.ndim == 1:
                            thetas32 = thetas32[idx]
                        else:
                            self._logger.error(
                                f"-- -- Thetas32 has wrong number of dimensions when applying model edition actions (s)")
                            sys.exit(1)
                    elif line_els[0] == 'd':
                        tpc = int(line_els[1])
                        ntopics = thetas32.shape[1]
                        tpc_keep = [k for k in range(ntopics) if k != tpc]
                        if thetas32.ndim == 2:
                            thetas32 = thetas32[:, tpc_keep]
                        elif thetas32.ndim == 1:
                            thetas32 = thetas32[tpc_keep]
                        else:
                            self._logger.error(
                                f"-- -- Thetas32 has wrong number of dimensions when applying model edition actions (s)")
                            sys.exit(1)
                    elif line_els[0] == 'f':
                        tpcs = [int(el) for el in line_els[1:]]
                        if thetas32.ndim == 2:
                            thet = np.sum(thetas32[:, tpcs], axis=1)
                            thetas32[:, tpcs[0]] = thet
                            thetas32 = np.delete(thetas32, tpcs[1:], 1)
                        elif thetas32.ndim == 1:
                            thet = np.sum(thetas32[tpcs], axis=0)
                            thetas32[tpcs[0]] = thet
                            thetas32 = np.delete(thetas32, tpcs[1:])
                        else:
                            self._logger.error(
                                f"-- -- Thetas32 has wrong number of dimensions when applying model edition actions (f)")
                            sys.exit(1)
        if thetas32.ndim == 2:
            thetas32 = normalize(thetas32, axis=1, norm='l1')
        elif thetas32.ndim == 1:
            thetas32 = normalize(thetas32.reshape(1, -1), axis=1, norm='l1')
        doc_topics_file_npy = infer_path.joinpath("doc-topics.npy")
        np.save(doc_topics_file_npy, thetas32)

        return thetas32

    @abstractmethod
    def predict(self):
        pass

    def get_final_thetas(
        self,
        thetas32: np.ndarray,
        thetas_thr: float,
        max_sum: int
    ) -> List[dict]:
        """
        Given the inferred document-topic proportions, it applies the model editions and returns the final thetas in the desired format

        Parameters
        ----------
        thetas32: np.ndarray
            Doc-topic distribution of the inferred documents
        thetas_thr: float
            Threshold for the inferred document-topic proportions (it should be the same used during training)
        max_sum: int
            Maximum sum of the topic proportions when attaining their string representation

        Returns
        -------
        List[dict]
            List of dictionaries with the inferred topics in string representation
        """

        # Apply same model editions made at the training stage
        thetas32 = self.apply_model_editions(thetas32)

        # Thresholding and normalization
        thetas32[thetas32 < thetas_thr] = 0
        if thetas32.ndim == 2:
            thetas32 = normalize(thetas32, axis=1, norm='l1')
        elif thetas32.ndim == 1:
            thetas32 = normalize(thetas32.reshape(1, -1), axis=1, norm='l1')

        # Transform thetas into string representation
        thetas32_rpr = self.transform_inference_output(thetas32, max_sum)

        return thetas32_rpr


class MalletInferencer(Inferencer):
    def __init__(self, logger=None):

        super().__init__(logger)

    def predict(
        self,
        inferConfigFile: pathlib.Path,
        mallet_path: pathlib.Path = None,
        max_sum: int = 1000,
        thetas_thr: float = 3e-3
    ) -> List[dict]:
        """
        Performs topic inference utilizing a pretrained model according to Mallet

        Parameters
        ----------
        inferConfigFile: pathlib.Path
            Path to the configuration file for the inference process
        mallet_path: pathlib.Path
            Path to the mallet binary
        max_sum: int
            Maximum sum of the topic proportions when attaining their string representation
        thetas_thr: float
            Threshold for the inferred document-topic proportions (it should be the same used during training)

        Returns
        -------
        List[dict]
            List of dictionaries with the inferred topics
        """

        with pathlib.Path(inferConfigFile).open('r', encoding='utf8') as fin:
            self._inferConfig = json.load(fin)

        # Check if the model to perform inference on exists
        model_for_inf = Path(
            self._inferConfig["model_for_infer_path"])
        if not os.path.isdir(model_for_inf):
            self._logger.error(
                f'-- -- Provided path for the model to perform inference on path is not valid -- Stop')
            return

        # A proper corpus should exist with the corresponding ipmortation pipe
        path_pipe = Path(
            self._inferConfig["model_for_infer_path"]).joinpath('modelFiles/import.pipe')
        if not path_pipe.is_file():
            self._logger.error(
                '-- Inference error. Importation pipeline not found')
            return

        # Holdout corpus should exist
        holdout_corpus = Path(
            self._inferConfig['infer_path']).joinpath("corpus.txt")
        if not holdout_corpus.is_file():
            self._logger.error(
                '-- Inference error. File to perform the inference on not found')
            return

        # Get inferencer
        inferencer = Path(
            self._inferConfig['model_for_infer_path']).joinpath("modelFiles/inferencer.mallet")

        # The following files will be generated in the same folder
        corpus_mallet_inf = \
            holdout_corpus.parent.joinpath('corpus_inf.mallet')
        doc_topics_file = holdout_corpus.parent.joinpath('doc-topics-inf.txt')

        # Get Mallet Path
        if mallet_path:
            mallet_path = Path(self.mallet_path)
        elif 'TMparam' in self._inferConfig.keys() and 'mallet_path' in self._inferConfig['TMparam'].keys():
            mallet_path = Path(self._inferConfig['TMparam']['mallet_path'])
        if not mallet_path.is_file():
            self._logger.error(
                f'-- -- Provided mallet path is not valid -- Stop')
            return

        # Import data to mallet
        self._logger.info('-- Inference: Mallet Data Import')

        cmd = mallet_path.as_posix() + \
            ' import-file --use-pipe-from %s --input %s --output %s'
        cmd = cmd % (path_pipe, holdout_corpus, corpus_mallet_inf)

        try:
            self._logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- Mallet failed to import data. Revise command')
            return

        # Get topic proportions
        self._logger.info('-- Inference: Inferring Topic Proportions')
        num_iterations = 100
        doc_topic_thr = 0

        cmd = mallet_path.as_posix() + \
            ' infer-topics --inferencer %s --input %s --output-doc-topics %s ' + \
            ' --doc-topics-threshold ' + str(doc_topic_thr) + \
            ' --num-iterations ' + str(num_iterations)
        cmd = cmd % (inferencer, corpus_mallet_inf, doc_topics_file)

        try:
            self._logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- Mallet inference failed. Revise command')
            return

        # Get inferred thetas
        ntopics = \
            self._inferConfig['TMparam']['ntopics']
        cols = [k for k in np.arange(2, ntopics + 2)]
        thetas32 = np.loadtxt(doc_topics_file, delimiter='\t',
                              dtype=np.float32, usecols=cols)

        thetas32_rpr = super().get_final_thetas(
            thetas32=thetas32,
            thetas_thr=thetas_thr,
            max_sum=max_sum)

        return thetas32_rpr


class SparkLDAInferencer(Inferencer):
    def __init__(self, logger=None):

        super().__init__(logger)

    def predict(self,
                inferConfigFile: pathlib.Path,
                max_sum: int = 1000) -> List[dict]:
        # TODO: Implement SparkLDAInferencer
        return


class ProdLDAInferencer(Inferencer):
    def __init__(self, logger=None):

        super().__init__(logger)

    def predict(
        self,
        inferConfigFile: pathlib.Path,
        max_sum: int = 100000,
        thetas_thr: float = 3e-3
    ) -> List[dict]:
        """
        Performs topic inference utilizing a pretrained model according to ProdLDA

        Parameters
        ----------
        inferConfigFile: pathlib.Path
            Path to the configuration file for the inference process
        max_sum: int
            Maximum sum of the topic proportions when attaining their string representation
        thetas_thr: float
            Threshold for the inferred document-topic proportions (it should be the same used during training)

        Returns
        -------
        List[dict]
            List of dictionaries with the inferred topics
        """
        with pathlib.Path(inferConfigFile).open('r', encoding='utf8') as fin:
            self._inferConfig = json.load(fin)

        # Check if the model to perform inference on exists
        model_for_inf = Path(
            self._inferConfig["model_for_infer_path"])
        if not os.path.isdir(model_for_inf):
            self._logger.error(
                f'-- -- Provided path for the model to perform inference on path is not valid -- Stop')
            return

        # A proper pickle file containing the avitm model should exist
        path_pickle = Path(
            self._inferConfig["model_for_infer_path"]).joinpath('modelFiles/model.pickle')
        if not path_pickle.is_file():
            self._logger.error(
                '-- Inference error. Pickle with the AVITM model not found')
            return

        # Holdout corpus should exist
        holdout_corpus = Path(
            self._inferConfig['infer_path']).joinpath("corpus.parquet")
        self._logger.info(
            f'-- Holdout corpus is {holdout_corpus}')
        if not os.path.isdir(holdout_corpus) and not os.path.isfile(holdout_corpus):
            self._logger.error(
                '-- Inference error. File to perform the inference on not found')
            return

        # Generating holdout corpus in the input format required by ProdLDA
        self._logger.info(
            '-- -- Inference: BOW Dataset object generation')
        df = pd.read_parquet(holdout_corpus)
        df_lemas = df[["bow_text"]].values.tolist()
        df_lemas = [doc[0].split() for doc in df_lemas]

        # Get avitm object for performing inference
        avitm = unpickler_avitm_for_case_inferencer(path_pickle, self._logger)

        # Prepare holdout corpus in avitm format
        ho_corpus = [el for el in df_lemas]
        ho_data = prepare_hold_out_dataset(
            ho_corpus, avitm.train_data.cv, avitm.train_data.idx2token)

        # Get inferred thetas matrix
        self._logger.info(
            '-- -- Inference: Getting inferred thetas matrix')
        thetas32 = np.asarray(
            avitm.get_doc_topic_distribution(ho_data))

        thetas32_rpr = super().get_final_thetas(
            thetas32=thetas32,
            thetas_thr=thetas_thr,
            max_sum=max_sum)

        return thetas32_rpr


class CTMInferencer(Inferencer):
    def __init__(self, logger=None):

        super().__init__(logger)

    def predict(
        self,
        inferConfigFile: pathlib.Path,
        max_sum: int = 100000,
        thetas_thr: float = 3e-3
    ) -> List[dict]:
        """
        Performs topic inference utilizing a pretrained model according to CTM

        Parameters
        ----------
        inferConfigFile: pathlib.Path
            Path to the configuration file for the inference process
        max_sum: int
            Maximum sum of the topic proportions when attaining their string representation
        thetas_thr: float
            Threshold for the inferred document-topic proportions (it should be the same used during training)

        Returns
        -------
        List[dict]
            List of dictionaries with the inferred topics
        """

        with pathlib.Path(inferConfigFile).open('r', encoding='utf8') as fin:
            self._inferConfig = json.load(fin)

        # Check if the model to perform inference on exists
        model_for_inf = Path(
            self._inferConfig["model_for_infer_path"])
        if not os.path.isdir(model_for_inf):
            self._logger.error(
                f'-- -- Provided path for the model to perform inference on path is not valid -- Stop')
            return

        # A proper pickle file containing the avitm model should exist
        path_pickle = Path(
            self._inferConfig["model_for_infer_path"]).joinpath('modelFiles/model.pickle')
        if not path_pickle.is_file():
            self._logger.error(
                '-- Inference error. Pickle with the CTM model not found')
            return

        # Get avitm object for performing inference
        ctm = unpickler(path_pickle)

        # Holdout corpus should exist
        holdout_corpus = Path(
            self._inferConfig['infer_path']).joinpath("corpus.parquet")
        if not os.path.isdir(holdout_corpus):
            self._logger.error(
                '-- Inference error. File to perform the inference on not found')
            return

        # Generating holdout corpus in the input format required by CTM
        self._logger.info(
            '-- -- Inference: CTM Dataset object generation')
        df = pd.read_parquet(holdout_corpus)
        df_lemas = df[["bow_text"]].values.tolist()
        df_lemas = [doc[0].split() for doc in df_lemas]
        corpus = [el for el in df_lemas]

        if not "embeddings" in list(df.columns.values):
            df_raw = df[["all_rawtext"]].values.tolist()
            df_raw = [doc[0].split() for doc in df_raw]
            unpreprocessed_corpus = [el for el in df_raw]
            embeddings = None
        else:
            embeddings = df.embeddings.values
            unpreprocessed_corpus = None

        ho_data = prepare_hold_out_dataset(
            hold_out_corpus=corpus,
            qt=ctm.train_data.qt, unpreprocessed_ho_corpus=unpreprocessed_corpus, embeddings_ho=embeddings)

        # Get inferred thetas matrix
        self._logger.info(
            '-- -- Inference: Getting inferred thetas matrix')
        thetas32 = np.asarray(
            ctm.get_doc_topic_distribution(ho_data))

        thetas32_rpr = super().get_final_thetas(
            thetas32=thetas32,
            thetas_thr=thetas_thr,
            max_sum=max_sum)

        return thetas32_rpr


##############################################################################
#                                  MAIN                                      #
##############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inference utilities')
    parser.add_argument('--config', type=str, default=None,
                        help="path to inference configuration file")
    parser.add_argument('--infer', action='store_true', default=False,
                        help="Perform inference according to config file")
    args = parser.parse_args()

    if args.infer:
        configFile = Path(args.config)
        if configFile.is_file():
            with configFile.open('r', encoding='utf8') as fin:
                infer_config = json.load(fin)

                if infer_config['trainer'] == 'mallet':
                    inferencer = MalletInferencer(infer_config)

                elif infer_config['trainer'] == 'sparkLDA':
                    inferencer = SparkLDAInferencer(infer_config)

                elif infer_config['trainer'] == 'prodLDA':
                    inferencer = ProdLDAInferencer(infer_config)

                elif infer_config['trainer'] == 'ctm':
                    inferencer = CTMInferencer(infer_config)

                inferencer.predict(configFile)
        else:
            sys.exit('You need to provide a valid configuration file')
