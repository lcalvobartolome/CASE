from flask_restx import Api

from .namespace_corpora import api as ns1
from .namespace_collections import api as ns2
from .namespace_models import api as ns3
from .namespace_queries import api as ns4

api = Api(
    title="CASE's Topic Modeling API",
    version='1.0',
    description='This RESTful API utilizes the Solr search engine for efficient data storage and retrieval of logical corpora and their associated topic models. Data is formatted according to the specifications provided by the topicmodeler, enabling seamless integration and search capabilities. The API also offers a range of query options to facilitate information retrieval.',
)

api.add_namespace(ns2, path='/collections')
api.add_namespace(ns1, path='/corpora')
api.add_namespace(ns3, path='/models')
api.add_namespace(ns4, path='/queries')