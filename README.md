# CASE

- [CASE](#case)
  - [Overview](#overview)
  - [Deployment Steps](#deployment-steps)
  - [Indexing](#indexing)
    - [Corpus Indexing](#corpus-indexing)
    - [Model Indexing](#model-indexing)
  - [Endpoints](#endpoints)
    - [Collections](#collections)
    - [Corpora](#corpora)
    - [Models](#models)
    - [Queries](#queries)

## Overview

CASE is a Solr-based exploitation tool designed to efficiently index metadata and topic information. It is optimized for calculating aggregated indicators, semantic similarities, and supporting web service requests.

The Solr-powered service is a multi-container application with a Solr search engine for data storage and retrieval. The Python-based RESTful API (`case-tm`) acts as an intermediary between Solr and the user (or frontend). It utilizes two additional services: `case-inferencer` for text inference using indexed models, and `case-classifier` for classification. The TM API also provides endpoints for indexing collections and topic models.

## Deployment Steps

1. **Prepare the Data Source**  
   Create a folder named `data/source` and place all the corpus and model information you wish to index into this directory.

2. **Create Docker Network**  
   Set up a Docker network named `case_net` using the following command:

   ```bash
   docker network create -d bridge case_net --subnet X.X.X.X/X --attachable
   ```

3. **Start the Services**  
   Launch the services using Docker Compose:

   ```bash
   docker-compose up -d
   ```

4. **Verify the Setup**  
   Ensure that the system is correctly initialized:
   - Access the Solr service, which should be available at [http://your_server_name:20003/solr/#/](http://your_server_name:20003/solr/#/).
   - Create a test collection using the `case_config` configset via the Solr interface. If the setup is successful, you may delete the test collection and proceed with indexing.

## Indexing

### Corpus Indexing

To index a corpus, we require the presence of the raw corpus in the mounted volume ``"/data/source"``.

Then, to index the HFRI corpus, named ``"HFRI.parquet"`` we do as depicted in the following image:

![Logical corpus indexing example](https://github.com/Nemesis1303/CASE/blob/main/static/Images/index_corpus.png)

This process creates a corpus collection named ``"hfri"`` in Solr. The collection includes all the metadata available in the parquet file. Additionally, it includes information related to the lemmas used for topic modeling calculations ("all_lemmas"). To maintain consistency among all the possible corpora indexed into the Solr collection, we rename the fields ``"id"``, ``"title"``, and ``"date"`` to these pseudonyms, regardless of their original names. The instructions for performing these field equivalences must be specified in the ``"case_config/config.cf"`` file prior to indexing. For more detailed information, you can refer to [here](https://github.com/Nemesis1303/CASE/tree/main/case_config/readme.md).

During corpus indexing, an entry is also created in the "corpora" collection. This collection stores information about all the corpus collections indexed in the Solr instances, along with their indexed models.

### Model Indexing

To index a model, the following requirements must be met:

1. The topic model entity should be present in the mounted volume ``"/data/source"``. This includes a folder named after the model, containing at least the ``"TMmodel"`` folder and the training configuration file (``"trainconfig.json"``).

2. The model to be indexed must be associated with a corpus that has already been indexed into the Solr instance.

To index a model (e.g., a model named ``"HFRI-30"``), follow the steps illustrated in the image below:

![Model indexing example](https://github.com/Nemesis1303/CASE/blob/main/static/Images/index_model.png)

This process creates a model collection named ``"hfri-30"`` in Solr. The collection includes all the metadata available in the model's ``"TMmodel"`` folder, namely word distribution, size, entropy, coherence, number of active documents, chemical description, labels, vocabulary, and coordinates in a 2D-space for each topic in the model.

Additionally, the corpus collection associated with the model is modified by adding two fields to each document that has a topical representation for that model:

- ``"doctpc_{model_name}"`` contains the document-topic distribution given by the model with the name "model_name".
- ``"sim_{model_name}"`` contains a list of the 50 most similar documents to the given document, according to the model with the name "model_name".
  
These additional fields are included in the corpus information within the ``"corpora"`` collection. Furthermore, the name of the model collection is added to the list of models associated with that corpus, as shown in the example below:

![Collection corpora after indexing corpus and model](https://github.com/Nemesis1303/CASE/blob/main/static/Images/after_model.png)

## Endpoints

### Collections

The endpoints in this category refer to generic Solr-related operations that, in principle, **will only be used internally:**

- ``/collections/createCollection/``: Creates a Solr collection.
- ``/collections/deleteCollection/``: Deletes a Solr collection.
- ``/collections/listCollections/``: List all collections available in the Solr instance.
- ``/collections/query/``: Performs a generic Solr query.

### Corpora

These endpoints performs corpora-related operations, that is, those related with the management, indexing and listing of linguistic data sets or collections known as corpora:

- ``/corpora/deleteCorpus/``: Deletes an entire corpus collection from the system.
- ``/corpora/indexCorpus/``: Indexes a corpus in a Solr collection, using the logical corpus name as the collection identifier.
- ``/corpora/listAllCorpus/``: Lists all available corpus collections in the Solr instance.
- ``/corpora/listCorpusModels/``: Lists all models associated with a specific corpus previously indexed in Solr.
- ``/corpora/listCorpusEWBdisplayed/``: Lists the corpus metadata fields that will be displayed in the EWB frontend.
- ``/corpora/listCorpusSearcheableFields/``: Lists the corpus metadata fields enabled for semantic search.
- ``/corpora/addSearcheableFields/``: Adds metadata fields to a corpus, enabling them for semantic search.
- ``/corpora/deleteSearcheableFields/``: Removes specific metadata fields from a corpus, disabling them from semantic search.

### Models

These endpoints performs models-related operations, that is, those related with the management, indexing and listing of topic models:

- ``/models/deleteModel/``: Deletes a model collection.
- ``/models/indexModel/``: Index the model information in a model collection and its corresponding corpus collection.
- ``/models/listAllModels/``: List all model collections available in the Solr instance.
- ``/models/addRelevantTpcForUser/``: Adds a topic's relevance information for a user to a model collection.
- ``/models/removeRelevantTpcForUser/``: Removes a topic's relevance information for a user to a model collection.

### Queries

|          **Endpoint**          |                                                                                             **Description**                                                                                             |                                                                                                                                                                                                                                    **Returns**                                                                                                                                                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| getThetasDocById               | Retrieve the document-topic distribution of a selected document in a corpus collection for a given topic model                                                                                          | ``{"thetas": thetas}``                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| getCorpusMetadataFields        | Get the available metadata fields for a specific corpus collection                                                                                                                                      | ``{"metadata_fields": meta_fields}``                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| getNrDocsColl                  | Get the number of documents in a collection                                                                                                                                                             | ``{"ndocs": ndocs}``                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| getDocsWithThetasLargerThanThr | Get documents with a topic proportion larger than a threshold  according to a selected topic model                                                                                                      | ``[{"id": id1, "doctpc_{model_name}": doctpc1 }, {"id": id2, "doctpc_{model_name}": doctpc2}, ...]``                                                                                                                                                                                                                                                                                                                                                                               |
| getDocsWithHighSimWithDocByid  | Retrieve documents that have a high semantic relationship with a selected document, i.e., its most similar documents                                                                                    | ``[{"id": id1, "score": score1 }, {"id": id2, "score": score2 }, ...]``                                                                                                                                                                                                                                                                                                                                                                                                            |
| getMetadataDocById             | Get the metadata of a selected document in a corpus collection                                                                                                                                          | ``{"metadata1": metadata1, "metadata2": metadata2, "metadata3": metadata3, ... }``                                                                                                                                                                                                                                                                                                                                                                                                 |
| getDocsWithString              | Retrieve the IDs of documents whose title contains a specific string in a corpus collection                                                                                                             | ``[{"id": id1}, {"id": id2}, ...]``                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| getTopicsLabels                | Get the labels associated with each topic in a given model                                                                                                                                              | ``[{"id": id1, "tpc_labels": label1 }, {"id": id2, "tpc_labels": label2}, ...]``                                                                                                                                                                                                                                                                                                                                                                                                   |
| getTopicTopDocs                | Get the top documents for a given topic in a model collection. Two criteria are considered: first, the thematic representation for the requested topic and second, the number of words in the document. | ``[{"id": id1, "thetas": thetas1, "num_words_per_doc": num_words_per_doc1 }, {"id": id2, thetas": thetas2, "num_words_per_doc": num_words_per_doc2}, ...]``                                                                                                                                                                                                                                                                                                                        |
| getModelInfo                   | Get information (chemical description, label, statistics, top docs, etc.) for each topic in a model collection                                                                                          | ``[{"id":id1, "betas": betas1, "alphas": alphas1, "topic_entropy":entropies1, "topic_coherence":cohrs1, "ndocs_active":active1, "tpc_descriptions":desc1, "tpc_labels":labels1, "coords":coords1, "top_words_betas":top_words_betas1,}, {"id":id2, "betas": betas2, "alphas": alphas2, "topic_entropy":entropies2, "topic_coherence":cohrs2, "ndocs_active":active2, "tpc_descriptions":desc2, "tpc_labels":labels2, "coords":coords2, "top_words_betas":top_words_betas2}, ...]`` |
| getBetasTopicById              | Get the word distribution of a selected topic in a model collection                                                                                                                                     | ``{"betas": betas}``                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| getMostCorrelatedTopics        | Get the most correlated topics to a given topic in a selected model                                                                                                                                     | ``[{"id": id1, "betas": betas1 }, {"id": id2, "betas": betas2}, ...]``                                                                                                                                                                                                                                                                                                                                                                                                             |
| getPairsOfDocsWithHighSim      | Retrieve pairs of documents with a semantic similarity larger than a certain threshold in a given topic model, filtered by year.                                                                        | ``[{"id_1": id1, "id_2": id2, "score": score1 }, {"id_1": id1, "id_2": id2, "score": score2}, ...]``                                                                                                                                                                                                                                                                                                                                                                               |
| getDocsSimilarToFreeText       | Get documents that are semantically similar to a free text according to a given topic model                                                                                                             | ``[{"id": id1, "score": score1 }, {"id": id2, "score": score2 }, ...]``                                                                                                                                                                                                                                                                                                                                                                                                            |
| getLemmasDocById               | Retrieve the lemmas of a selected document in a corpus collection                                                                                                                                       | ``{"thetas": thetas}``                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| getThetasAndDateAllDocs        | Get the date and document-topic representation associated with a given model for all documents in a corpus collection                                                                                   | ``[{"id": id1, "date": date1, "doctpc_{model_name}":doctpc1}, {"id": id2, "date": date2, "doctpc_{model_name}":doctpc2}, ...]``                                                                                                                                                                                                                                                                                                                                                    |
| getBetasByWordAndTopicId       | Get the topic-word distribution of a given word in a given topic associated with a given model                                                                                                          | ``{"betas": betas}``                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| getBOWbyDocsIDs                | Get the BoW counts of a given words in a document.                                                                                                                                                      | ``{"id": id, "payload(bow,word)": count}``                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| getUserRelevantTopics          | Get the topic-word distribution of a given word in a given topic associated with a given model                                                                                                          | ``[{"id":id1, "betas": betas1, "alphas": alphas1, "topic_entropy":entropies1, "topic_coherence":cohrs1, "ndocs_active":active1, "tpc_descriptions":desc1, "tpc_labels":labels1, "coords":coords1, "top_words_betas":top_words_betas1,}, {"id":id2, "betas": betas2, "alphas": alphas2, "topic_entropy":entropies2, "topic_coherence":cohrs2, "ndocs_active":active2, "tpc_descriptions":desc2, "tpc_labels":labels2, "coords":coords2, "top_words_betas":top_words_betas2}, ...]`` |