import argparse
import logging
import sys
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
import time
import pathlib

def calculate_s3(
    logger: logging.Logger,
    tm_model_dir:str
):
    """Given the path to a TMmodel, it calculates the similarities between documents and saves them in a sparse matrix.

    Parameters
    ----------
    logger : logging.Logger
        Logger.
    tm_model_dir : str
        Path to TMmodel.
    """
    
    t_start = time.perf_counter()
    TMfolder = pathlib.Path(tm_model_dir)
    thetas = sparse.load_npz(TMfolder.joinpath('thetas.npz')).todense()
    betas = np.load(TMfolder.joinpath('betas.npy'))
    vocab_w2id = {}
    with TMfolder.joinpath('vocab.txt').open('r', encoding='utf8') as fin:    
        for i, line in enumerate(fin):
            wd = line.strip()
            vocab_w2id[wd] = i
    
    logger.info(f"Shape of thetas: {np.shape(thetas)} ")
    logger.info(f"Shape of betas: {np.shape(betas)} ")
    
    corpusFile = TMfolder.parent.joinpath('corpus.txt')
    with corpusFile.open("r", encoding="utf-8") as f:
        lines = [line for line in f.readlines()]
    documents_texts = [line.rsplit(" 0 ")[1].strip().split() for line in lines if line.rsplit(" 0 ")[1].strip().split() != []]
    
    D = len(thetas)
    K = len(betas)
    S3 = np.zeros((D, K))

    for doc in range(D):
        for topic in range(K):
            wd_ids = [
                vocab_w2id[word] 
                for word in documents_texts[doc] 
                if word in vocab_w2id
            ]
            S3[doc, topic] = np.sum(betas[topic, wd_ids])
            
    sparse_S3 = csr_matrix(S3)
    sparse.save_npz(TMfolder.joinpath('s3.npz'), sparse_S3)

    t_end = time.perf_counter()
    t_total = (t_end - t_start)/60
    logger.info(f"Total computation time: {t_total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_tmmodel', type=str, default="/export/usuarios_ml4ds/lbartolome/Repos/my_repos/CASE/data/HFRI-30/TMmodel", help="Path to TMmodel.")
    
    ################### LOGGER #################
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    ############################################
    
    args = parser.parse_args()
    
    calculate_s3(logger, args.path_tmmodel)

if __name__ == '__main__':
    main()