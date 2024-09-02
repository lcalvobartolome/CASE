# Similarities Calculation

This folder contains auxiliary scripts to:

1. Calculate the similarities between each pair of documents in the corpus.
2. Save the obtained similarities in text format for faster indexing into the Solr engine.

For calculating the similarities, we utilize the [``sparse_dot_topn``](https://github.com/ing-bank/sparse_dot_topn) library, which enables fast sparse matrix multiplication. Specifically, the function ``awesome_cossim_topn`` efficiently computes the cosine similarity between the document-topic distributions (thetas) of the documents in a corpus, represented as sparse matrices, and returns the top-N most similar pairs.

Like any other part of the EWB, these scripts assume that topic models are provided in the format of the ``topicmodeler``, where all topic modeling outputs are encapsulated within a ``TMmodel`` folder. Note that the execution the ``similarities.py`` script is only necessary if the similarities have not been constructed during the creation of the TMmodel (this option can be activated or deactivated in the topicmodeler).

Configuring the parameters for the ``awesome_cossim_topn`` function can be a bit tricky. It accepts the following parameters, among others:

* `ntop`: the number of top-N pairs of most similar rows to be returned.
* `lower_bound`: a threshold similarity score; it returns only those pairs that exceed this threshold.

The higher the value of `ntop`, the more time-consuming the computation becomes, and it increases the time required to index the similarities into the Solr engine. Furthermore, after conducting experiments with the Cordis datasets (approximately 60,000 documents), it was determined that the maximum value of `ntop` that allowed for correct indexing and functioning of subsequent queries was `ntop=10,000`. This provided a lower bound for the similarity calculation of 40%. However, for our project, which focuses on using similarity search for plagiarism detection and related activities, we do not require such a stringent lower bound. Therefore, we found that using `ntop=300` for the Cordis calculation was sufficient. However, this value may vary for larger corpora. As a result, a potential solution is to set the `lower_bound` to 60 % (0.6) and experiment with different values of `ntop` for larger datasets than CORDIS.