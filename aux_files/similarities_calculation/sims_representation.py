import argparse
import logging
import pathlib
import sys
import scipy.sparse as sparse
from typing import List
import time

def get_doc_by_doc_sims(W, ids_corpus) -> List[str]:
    """
    Calculates the similarity between each pair of documents in the corpus collection based on the document-topic distribution provided by the model being indexed.

    Parameters
    ----------
    W: scipy.sparse.csr_matrix
        Sparse matrix with the similarities between each pair of documents in the corpus collection.
    ids_corpus: List[str]
        List of ids of the documents in the corpus collection.

    Returns:
    --------
    sims: List[str]
        List of string represenation of the top similarities between each pair of documents in the corpus collection.
    """

    # Get the non-zero elements indices
    non_zero_indices = sparse.triu(W, k=1).nonzero()

    # Convert to a string
    sim_str = \
        [' '.join([f"{ids_corpus[col]}|{W[row, col]}" for col in non_zero_indices[1]
                    [non_zero_indices[0] == row]][1:]) for row in range(W.shape[0])]

    return sim_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_tmmodel', type=str,
                        default="/export/data_ml4ds/IntelComp/CASE/data/source/Mallet-30/TMmodel",
                        help="Path to TMmodel.")
    
    ################### LOGGER #################
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    ############################################
    
    args = parser.parse_args()
    
    # Calculate similarities string representation
    sims = sparse.load_npz(pathlib.Path(args.path_tmmodel).joinpath("distances.npz"))
    logger.info(f"Sims obtained")

    def process_line(line):
        id_ = line.rsplit(' 0 ')[0].strip()
        id_ = int(id_.strip('"'))
        return id_

    with open(pathlib.Path(args.path_tmmodel).parent.joinpath("corpus.txt"), encoding="utf-8") as file:
        ids_corpus = [process_line(line) for line in file]
    logger.info(f"Ids obtained")
    logger.info(f"Starting similarities representation...")
    time_start = time.perf_counter()
    sim_rpr = get_doc_by_doc_sims(sims, ids_corpus)
    time_end = time.perf_counter()
    logger.info(f"Similarities representation finished in {time_end - time_start:0.4f} seconds")
    logger.info(f"Writing similarities representation to txt file...")

    # Save similarities representation to txt file
    with open(pathlib.Path(args.path_tmmodel).joinpath('distances.txt'), 'w') as f:
        for item in sim_rpr:
            f.write("%s\n" % item)
    
if __name__ == '__main__':
    main()