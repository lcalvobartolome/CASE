
"""
This module provides 4 classes, each of them extending each of the inferencers provided at the '/base/inferencer.py' module. By doing so, it pass specific configuration parameters specifc to the CASE and returns the inferencer's response in the format expected by the API.

Author: Lorena Calvo-Bartolomé
Date: 19/05/2023
"""

import configparser
import logging
import pathlib
import time
from typing import Union

from src.core.inferencer.base.inferencer import (CTMInferencer,
                                                 MalletInferencer,
                                                 ProdLDAInferencer,
                                                 SparkLDAInferencer)


class CASEMalletInferencer(MalletInferencer):
    def __init__(self,
                 logger: logging.Logger,
                 config_file: str = "/config/config.cf") -> None:
        """
        Initilization Method

        Parameters
        ----------
        logger: Logger object
            To log object activity
        config_file: str
            Path to the config file
        """

        super().__init__(logger)

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self.mallet_path = cf.get('mallet', 'mallet_path')
        self.max_sum = cf.getint('restapi', 'thetas_max_sum')
        self.thetas_thr = cf.getfloat('inferencer', 'thetas_thr')

        return

    def predict(self, inferConfigFile: pathlib.Path) -> Union[dict, None]:
        """Execute inference on the given text and returns a response in the format expected by the API.

        Parameters
        ----------
        inferConfigFile: pathlib.Path
            Path to the configuration file for inference

        Returns
        -------
        response: dict
            A dictionary containing the response header and the response, if any, following the API format:
            {
                "responseHeader": {
                    "status": 200 or 400,
                    "time": 0.0
                },  
                "response": {
                    [{'id': 1, thetas: 't1|X1 t2|X1 t3|X3 t4|X4 t5|X5'},
                     {'id': 1, thetas: 't1|X1 t5|X5'},
                     {'id': 1, thetas: 't2|X2 t4|X4 t5|X5'},
                     ...] or None
                }
            }     
        sc: int
            Status code of the response                                               
        """

        start_time = time.time()

        try:
            resp = super().predict(inferConfigFile=inferConfigFile,
                                   mallet_path=self.mallet_path,
                                   max_sum=self.max_sum,
                                   thetas_thr=self.thetas_thr)
            end_time = time.time() - start_time
            sc = 200
            responseHeader = {"status": sc,
                              "time": end_time}

            response = {"responseHeader": responseHeader,
                        "response": resp}

            self._logger.info(f"-- -- Inference completed successfully")

        except Exception as e:
            end_time = time.time() - start_time
            sc = 400
            responseHeader = {"status": sc,
                              "time": end_time,
                              "error": str(e)}

            response = {"responseHeader": responseHeader,
                        "response": None}

            self._logger.info(
                f"-- -- Inference failed with error: {str(e)}")

        return response, sc


class CASESparkLDAInferencer(SparkLDAInferencer):
    def __init__(self,
                 logger: logging.Logger,
                 config_file: str = "/config/config.cf") -> None:
        """
        Initilization Method

        Parameters
        ----------
        logger: Logger object
            To log object activity
        config_file: str
            Path to the config file
        """

        super().__init__(logger)

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self.max_sum = cf.getint('restapi', 'thetas_max_sum')
        self.thetas_thr = cf.getfloat('inferencer', 'thetas_thr')
        #CHECK if necessary use max_sum_neural_models

        return

    def predict(self, inferConfigFile: pathlib.Path) -> Union[dict, None]:
        # TODO: Implement predict method
        pass


class CASEProdLDAInferencer(ProdLDAInferencer):
    def __init__(self,
                 logger: logging.Logger,
                 config_file: str = "/config/config.cf") -> None:
        """
        Initilization Method

        Parameters
        ----------
        logger: Logger object
            To log object activity
        config_file: str
            Path to the config file
        """

        super().__init__(logger)

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self.max_sum = cf.getint('restapi', 'max_sum_neural_models')
        self.thetas_thr = cf.getfloat('inferencer', 'thetas_thr')

        return

    def predict(self, inferConfigFile: pathlib.Path) -> Union[dict, None]:
        """Execute inference on the given text and returns a response in the format expected by the API.

        Parameters
        ----------
        inferConfigFile: pathlib.Path
            Path to the configuration file for inference

        Returns
        -------
        response: dict
            A dictionary containing the response header and the response, if any, following the API format:
            {
                "responseHeader": {
                    "status": 200 or 400,
                    "time": 0.0
                },  
                "response": {
                    [{'id': 1, thetas: 't1|X1 t2|X1 t3|X3 t4|X4 t5|X5'},
                     {'id': 1, thetas: 't1|X1 t5|X5'},
                     {'id': 1, thetas: 't2|X2 t4|X4 t5|X5'},
                     ...] or None
                }
            }     
        sc: int
            Status code of the response                                               
        """

        start_time = time.time()

        try:
            resp = super().predict(inferConfigFile=inferConfigFile,
                                   max_sum=self.max_sum)
            end_time = time.time() - start_time
            sc = 200
            responseHeader = {"status": sc,
                              "time": end_time}

            response = {"responseHeader": responseHeader,
                        "response": resp}

            self._logger.info(f"-- -- Inference completed successfully")

        except Exception as e:
            end_time = time.time() - start_time
            sc = 400
            responseHeader = {"status": sc,
                              "time": end_time,
                              "error": str(e)}

            response = {"responseHeader": responseHeader,
                        "response": None}

            self._logger.info(
                f"-- -- Inference failed with error: {str(e)}")

        return response, sc
    

class CASECTMInferencer(CTMInferencer):
    def __init__(self,
                 logger: logging.Logger,
                 config_file: str = "/config/config.cf") -> None:
        """
        Initilization Method

        Parameters
        ----------
        logger: Logger object
            To log object activity
        config_file: str
            Path to the config file
        """

        super().__init__(logger)

        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        self.max_sum = cf.getint('restapi', 'max_sum_neural_models')
        self.thetas_thr = cf.getfloat('inferencer', 'thetas_thr')

        return

    def predict(self, inferConfigFile: pathlib.Path) -> Union[dict, None]:
        """Execute inference on the given text and returns a response in the format expected by the API.

        Parameters
        ----------
        inferConfigFile: pathlib.Path
            Path to the configuration file for inference

        Returns
        -------
        response: dict
            A dictionary containing the response header and the response, if any, following the API format:
            {
                "responseHeader": {
                    "status": 200 or 400,
                    "time": 0.0
                },  
                "response": {
                    [{'id': 1, thetas: 't1|X1 t2|X1 t3|X3 t4|X4 t5|X5'},
                     {'id': 1, thetas: 't1|X1 t5|X5'},
                     {'id': 1, thetas: 't2|X2 t4|X4 t5|X5'},
                     ...] or None
                }
            }     
        sc: int
            Status code of the response                                               
        """

        start_time = time.time()

        try:
            resp = super().predict(inferConfigFile=inferConfigFile,
                                   max_sum=self.max_sum)
            end_time = time.time() - start_time
            sc = 200
            responseHeader = {"status": sc,
                              "time": end_time}

            response = {"responseHeader": responseHeader,
                        "response": resp}

            self._logger.info(f"-- -- Inference completed successfully")

        except Exception as e:
            end_time = time.time() - start_time
            sc = 400
            responseHeader = {"status": sc,
                              "time": end_time,
                              "error": str(e)}

            response = {"responseHeader": responseHeader,
                        "response": None}

            self._logger.info(
                f"-- -- Inference failed with error: {str(e)}")

        return response, sc
    