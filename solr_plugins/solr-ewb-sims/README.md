# solr-ewb-sims

This code implements a Java plugin for Solr to retrieve pairs of documents with a semantic similarity larger than a certain threshold in a given topic model.

> This plugin is a modification of the ["Vector Scoring Plugin for Solr : Dot Product and Cosine Similarity"](https://github.com/saaay71/solr-vector-scoring) to be adequate for Solr 9 and the purpose stated above.

For it to work, similarities among each pair of documents for a given topic model (``model_name``) should be indexed in each document in a corpus collection using payload notation (i.e., ``1312|99.99 23453|98.7...``) in the field ``sim_{model_name}``.

Here, for each document ``id_i``, each component ``id_x|score_x`` within its ``sim_{model_name}`` field corresponds to the similarity among the document ``id_i`` and ``id_x``.

Considering the plugin has been correctly added to Solr, the query can be used as follows:

```bash
{!vs f=sim_mallet-10 vector="80,90"}
```

This query launches the Solr plugin and searches the ``sim_mallet-10`` field of each document in the Cordis collection for documents with a similarity between 80% and 90%. For each document, it returns a double, which indicates the range of those documents that are between this similarity percentage.

Let's consider that for the document with ``id = 83456``, its ``sim_mallet-10`` field is:

``sim_mallet-10 = 1312|99.99 23453|98.7 34534|96.5 34534|94.5 4564|93.75 45645|92.24 785|91.14 6876|89.88 45645|88.15 4564|85.47 53456|83.12 35645|82.08 4564|81.45 13563|80.12 13563|78.92….``

If the Solr query returns the score ``7.13``, this means that it will keep the documents between positions 7 and 13 that are within this range. In the example they would be:

``6876|89.88 45645|88.15 4564|85.47 53456|83.12 35645|82.08 4564|81.45 13563|80.12``

