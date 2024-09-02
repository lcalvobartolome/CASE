
"""
This module is a wrapper around the clf-inference-intelcomp module (Inference system for hierarchical text classification) to use within the CASE project.

Author: Lorena Calvo-Bartolomé
Date: 07/09/2023
"""

import configparser
import logging
from typing import Union
from clf_inference_intelcomp import Classification
import torch


class CASEClassifier():
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
        
        self._logger = logger
                
        # Read configuration from config file
        cf = configparser.ConfigParser()
        cf.read(config_file)
        
        # Initialize the classifier
        self._classifier = Classification()
        self._classifier.CACHE_DIR = cf.get('classifier', 'cache_dir')
        self._classifier.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        return
    
    def classify(self, text: Union[str, list[str]], taxonomy:str='ipc'):
        """Classifies the given text using models trained in the given taxonomy.
        If text is a list, it performs text classification by batches.
        
        Parameters
        ----------
        text: str or list[str]
            Text to classify
        taxonomy: str
            Taxonomy to use for classification. Possible values are: 'ipc', 'fos', 'nace2'
        
        Returns:
        -------
        dict / list[dict]
            A dictionary/list of dictionaries, with each containing the result obtained at a different level of the taxonomy:
            - The key indicates the level of depth (0,1,2...) in the classification tree.
            - The value is a triplet that includes the class code, class name and confidence score.
        """
        
        if type(text) == list:
            return self._classifier.classify_batch(taxonomy, text)
        else:
            return self._classifier.classify(taxonomy, text)
    
    def get_avail_taxonomies(self):
        """Get the list of available taxonomies.

        Returns:
        -------
        list[str]
            List of available taxonomies.
        """
        
        return self._classifier.avail_taxonomies
    
    def cache_models(self, taxonomy: str = None) -> None:
        """Caches languages models in memory to avoid having to download them at inference time.
        
        Parameters
        ----------
        taxonomy : str
            Can be either the name of one of the available taxonomies (so that only its models are loaded in memory) or the keyword 'all' (default value) that allows caching all models at once.
        """
        
        self._classifier.cache_models(taxonomy) 
        return
    
    def list_models(self) -> dict:
        """Lists the models to be used for each taxonomy, with a different indentation for each of its levels.
        The first level (i.e. 0) of any taxonomy only contains a single model, which would be the root node of the classification tree, while deeper levels have a list of models for each possible outcome from the prior level. This is why, for the non-zero levels, it is important to make sure that the model key (value preceding ":") matches some label from the previous model in the chain.
        
        Returns
        -------
        dict
            Dictionary with the paths to the different models, hierarchically organized.
        """
        return self._classifier.models
    
    def list_classes(self) -> dict:
        """Lists the possible outputs of all classifiers by mapping class codes to their corresponding names.
        
        Returns
        -------
        dict
            Dictionary to map class codes to class names, hierarchically organized as well.
        """
        
        return self._classifier.classes