from flask_restx import Api

from .namespace_classify import api as ns1

api = Api(
    title="CASE's Classifier API",
    version='1.0',
    description='A RESTful API designed to classify documents based on a given hierarchy of language models.',
)

api.add_namespace(ns1, path='/classification')