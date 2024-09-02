"""
This  module provides 2 classes to handle Inferencer API responses and requests.

The InferencerResponse class handles Inferencer API response and errors, while the CASEInferencerClient class handles requests to the Inferencer API.

Author: Lorena Calvo-Bartolomé
Date: 21/05/2023
"""

import logging
import os

import requests
from src.core.clients.external.api_generic.client import Client
from src.core.clients.external.api_generic.response import Response


class InferencerResponse(Response):
    """
    A class to handle Inferencer API response and errors.
    """

    def __init__(self,
                 resp: requests.Response,
                 logger: logging.Logger) -> None:

        super().__init__(resp, logger)
        return


class CASEInferencerClient(Client):
    """
    A class to handle CASE Inferencer API requests.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Parameters
        ----------
        logger : logging.Logger
            The logger object to log messages and errors.
        """

        super().__init__(logger, "Inferencer")
        
        # Get the Inferencer URL from the environment variables
        self.inferencer_url = os.environ.get('INFERENCE_URL')

        # Initialize requests session and logger
        self.inferencer = requests.Session()
        
        return

    def _do_request(self,
                    type: str,
                    url: str,
                    timeout: int = 10,
                    **params) -> InferencerResponse:
        """Sends a request to the Inferencer API and returns an object of the InferencerResponse class.

        Parameters
        ----------
        type : str
            The type of the request.
        url : str
            The URL of the Inferencer API.
        timeout : int, optional
            The timeout of the request in seconds, by default 10.
        **params: dict
            The parameters of the request.

        Returns
        -------
        InferencerResponse: InferencerResponse
            An object of the InferencerResponse class.
        """

        # Send request
        resp = super()._do_request(type, url, timeout, **params)
        
        # Parse Inference response
        inf_resp = InferencerResponse(resp, self.logger)

        return inf_resp

    def infer_doc(self,
                  text_to_infer: str,
                  model_for_inference: str) -> InferencerResponse:
        """Execute inference on the given text and returns a response in the format expected by the API.

        Parameters
        ----------
        text_to_infer : str
            The text to infer.
        model_for_inference : str
            The model to use for inference.

        Returns
        -------
        InferencerResponse: InferencerResponse
            An object of the InferencerResponse class.
        """

        headers_ = {'Accept': 'application/json'}

        params_ = {
            'text_to_infer': text_to_infer,
            'model_for_infer': model_for_inference
        }

        url_ = '{}/inference_operations/inferDoc'.format(self.inferencer_url)

        # Send request to Inferencer
        inf_resp = self._do_request(
            type="get", url=url_, timeout=120, headers=headers_, params=params_)

        return inf_resp

    def infer_corpus(self,
                     corpus_to_infer: str,
                     model_for_inference: str):
        # TODO: Implement this method
        pass
