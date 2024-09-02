"""
Some utility functions.

Author: Lorena Calvo-Bartolomé
Date: 19/05/2023
"""

import datetime as DT
import json
import os
import pathlib
import pickle
import random
import shutil
from typing import Union

import numpy as np
import pandas as pd

import logging

from src.core.models.neural_models.pytorchavitm.avitm_network.avitm import \
    AVITM
from src.core.models.neural_models.pytorchavitm.avitm_network.decoder_network import \
    DecoderNetwork
from src.core.models.neural_models.pytorchavitm.avitm_network.inference_network import \
    InferenceNetwork
from src.core.models.neural_models.pytorchavitm.datasets.bow_dataset import \
    BOWDataset
from src.core.models.neural_models.utils.early_stopping.pytorchtools import \
    EarlyStopping


def unpickler(file: str):
    """Unpickle file"""
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickler(file: str, ob):
    """Pickle object to file"""
    with open(file, 'wb') as f:
        pickle.dump(ob, f)
    return 0

def unpickler_avitm_for_case_inferencer(path_pickle: pathlib.Path, logger: logging.Logger):
    """
    Unpickle the AVITM model for the CASE Inferencer

    Parameters
    ----------
    path_pickle: pathlib.Path
        Path to the pickle file
    
    Returns
    -------
    avitm_model: AVITM
        AVITM model
    """
    all_data = unpickler(path_pickle)
    train_data = BOWDataset(**all_data[0])
    validation_data = BOWDataset(**all_data[1])
    early_stopping = EarlyStopping()
    early_stopping.__dict__ = all_data[2]
    inf_net = InferenceNetwork(
        input_size=all_data[5]['input_size'],
        output_size=all_data[5]['output_size'],
        hidden_sizes=all_data[5]['hidden_sizes'])
    inf_net.__dict__ = all_data[5]

    decoder = DecoderNetwork(input_size=all_data[4]['input_size'])
    all_data[4]["_modules"]["inf_net"] = inf_net
    decoder.__dict__ = all_data[4]

    avitm = AVITM(logger=logger, input_size=all_data[3]['input_size'])
    all_data[3]['model'] = decoder
    all_data[3]['early_stopping'] = early_stopping
    all_data[3]['train_data'] = train_data
    all_data[3]['validation_data'] = validation_data
    avitm.__dict__ = all_data[3]

    return avitm


def sum_up_to(vector: np.ndarray, max_sum: int) -> np.ndarray:
    """It takes in a vector and a max_sum value and returns a NumPy array with the same shape as vector but with the values adjusted such that their sum is equal to max_sum.

    Parameters
    ----------
    vector: 
        The vector to be adjusted.
    max_sum: int
        Number representing the maximum sum of the vector elements.

    Returns:
    --------
    x: np.ndarray
        A NumPy array of the same shape as vector but with the values adjusted such that their sum is equal to max_sum.
    """
    x = np.array(list(map(np.int_, vector*max_sum))).ravel()
    pos_idx = list(np.where(x != 0)[0])
    while np.sum(x) != max_sum:
        idx = random.choice(pos_idx)
        x[idx] += 1
    return x


def find_folder(path: str, target_folder: str):
    """Find folder in path by name.

    Parameters
    ----------
    path: str
        Path to search for the folder.
    target_folder: str
        Name of the folder to be found.

    Returns
    -------
    str or None if not found
    """
    target_folder_lower = target_folder.lower()
    for directory_name in os.listdir(path):
        if directory_name.lower() == target_folder_lower and os.path.isdir(os.path.join(path, directory_name)):
            return os.path.join(path, directory_name)
    return None


def get_infer_config(logger: logging.Logger,
                     text_to_infer: str,
                     model_for_infer: str,
                     path_to_source: str = "/data/source",
                     path_to_infer: str = "/data/inference") -> Union[pathlib.Path, str]:
    """Get the configuration for the inference of a text.

    Parameters
    ----------
    text_to_infer: str
        Text to be inferred.
    model_for_infer: str
        Name of the model to be used for the inference.
    path_to_source: str
        Path to the source folder.
    path_to_infer: str
        Path to the inference folder.

    Returns
    -------
    outfile_infer: pathlib.Path
        Path to the inference config file.
    trainer: str
        Name of the trainer used for the inference.
    """

    description = f"Inference of text '{text_to_infer}'."

    # Find the location of the model to infer
    model_for_infer_path = find_folder(path=path_to_source,
                                       target_folder=model_for_infer)

    creation_date = DT.datetime.now().strftime('%Y%m%d')
    infer_path = pathlib.Path(path_to_infer).joinpath(creation_date)

    if infer_path.exists():
        # Remove current backup folder, if it exists
        old_model_dir = pathlib.Path(str(infer_path) + '_old/')
        if old_model_dir.exists():
            shutil.rmtree(old_model_dir)

        # Copy current model folder to the backup folder.
        shutil.move(infer_path, old_model_dir)
        print(
            f'-- -- Creating backup of existing inference model in {old_model_dir}')
    infer_path.mkdir()

    tr_config = \
        pathlib.Path(model_for_infer_path).joinpath("trainconfig.json")
    with tr_config.open('r', encoding='utf8') as fin:
        tr_config = json.load(fin)

    Preproc = tr_config['Preproc']
    trainer = tr_config['trainer']
    TMparam = tr_config['TMparam']

    # Delete file if it exists
    path_to_trset = \
        infer_path.joinpath(
            "corpus.txt") if trainer == 'mallet' else infer_path.joinpath("corpus.parquet")
    if path_to_trset.is_file():
        path_to_trset.unlink()
    elif path_to_trset.is_dir():
        shutil.rmtree(path_to_trset)

    # Save text to infer in a temporary file according to trainer format
    if trainer == 'mallet':
        with open(path_to_trset, 'w', encoding='utf-8') as fout:
            fout.write(
                str(1) + ' 0 ' + text_to_infer + '\n')
    else:
        # TODO: Add embeddings so it works for neural models
        if trainer == 'ctm': 
            pd.DataFrame(
            [[0, text_to_infer]], columns=["id", "bow_text", "embeddings"]
            ).to_parquet(path_to_trset)

        pd.DataFrame(
            [[0, text_to_infer]], columns=["id", "bow_text"]
        ).to_parquet(path_to_trset)

    # Create inference config
    infer_config = {
        "description": description,
        "infer_path": infer_path.as_posix(),
        "model_for_infer_path": model_for_infer_path,
        "trainer": trainer,
        "TrDtSet": path_to_trset.as_posix(),
        "text_to_infer": text_to_infer,
        "Preproc": Preproc,
        "TMparam": TMparam,
        "creation_date": creation_date,
    }

    outfile_infer = infer_path.joinpath("config.json")
    with outfile_infer.open('w', encoding='utf-8') as outfile:
        json.dump(infer_config, outfile,
                  ensure_ascii=False, indent=2, default=str)

    return outfile_infer, trainer
