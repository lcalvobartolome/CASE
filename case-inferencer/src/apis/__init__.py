from flask_restx import Api

from .namespace_infer import api as ns1

api = Api(
    title="CASE's Inferencer API",
    version='1.0',
    description='A RESTful API designed to infer the document-topic representation of a given text using a specified topic model. It provides a convenient way to analyze text documents and extract topic distributions.',
)

api.add_namespace(ns1, path='/inference_operations')
