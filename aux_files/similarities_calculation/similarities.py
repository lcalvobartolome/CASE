import argparse
import logging
import sys
import numpy as np
import scipy.sparse as sparse
from sparse_dot_topn import awesome_cossim_topn
import time
import pathlib



def calculate_sims(logger: logging.Logger,
                   tm_model_dir:str,
                   topn:int=300,
                   lb:float=0):
  """Given the path to a TMmodel, it calculates the similarities between documents and saves them in a sparse matrix.

  Parameters
  ----------
  logger : logging.Logger
      Logger.
  tm_model_dir : str
      Path to TMmodel.
  topn : int, optional
      Number of top similar documents to be saved. The default is 300.
  lb : float, optional
      Lower bound for the similarity. The default is 0.6.
  """
  t_start = time.perf_counter()
  TMfolder = pathlib.Path(tm_model_dir)
  thetas = sparse.load_npz(TMfolder.joinpath('thetas.npz'))
  logger.info(f"Shape of thetas: {np.shape(thetas)} ")
  thetas_sqrt = np.sqrt(thetas)
  thetas_col = thetas_sqrt.T
  
  logger.info(f"Topn: {topn}")
  sims = awesome_cossim_topn(thetas_sqrt, thetas_col, topn, lb)
  sparse.save_npz(TMfolder.joinpath('distances.npz'), sims)

  t_end = time.perf_counter()
  t_total = (t_end - t_start)/60
  logger.info(f"Total computation time: {t_total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_tmmodel', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/CORDIS/case_models/root_model_30_tpcs_20231028/TMmodel",
                        help="Path to TMmodel.")
    parser.add_argument('--topn', type=int, default=300,
                        help="Number of top similar documents to be saved.")
    parser.add_argument('--lb', type=float, default=0,
                        help="Lower bound for the similarity.")
    
    ################### LOGGER #################
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    ############################################
    
    args = parser.parse_args()
    
    # Calculate similarities
    calculate_sims(logger, args.path_tmmodel, args.topn, args.lb)

if __name__ == '__main__':
    main()