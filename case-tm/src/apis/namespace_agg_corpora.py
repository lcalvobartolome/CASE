"""
This script defines a Flask RESTful namespace for managing 'Aggregated corpora' (researchers & research group) stored in Solr as independent collections. 

Author: Lorena Calvo-Bartolomé
Date: 15/11/2024
"""

from flask_restx import Namespace, Resource, fields, reqparse # type: ignore
from src.core.clients.case_solr_client import CASESolrClient

# ======================================================
# Define namespace for managing corpora
# ======================================================
api = Namespace(
    'Aggregated corpora', description='Operations related with Aggregated corpora (researchers and research group).')

# ======================================================
# Namespace variables
# ======================================================
# Create Solr client
sc = CASESolrClient(api.logger)

# ======================================================
# Endpoints
# ======================================================
index_parser = reqparse.RequestParser()
index_parser.add_argument(
    "aggregated_corpus_name", help="Name of the aggregated corpus to index. For example, if the aggregated corpus we want to index is stored in a file name 'aggregated_corpus.parquet', the aggregated_corpus_name should be 'aggregated_corpus' (without the extension quotes). Do not use quotes in the name of the aggregated corpus.")
index_parser.add_argument(
    "aggregated_corpus_type", help="Type of the aggregated corpus to index, either 'researcher' or 'research_group'.")

@api.route('/indexAggregatedCorpus/')
class indexAggregatedCorpus(Resource):
    @api.doc(parser=index_parser)
    def post(self):
        args = index_parser.parse_args()
        aggregated_corpus_name = args['aggregated_corpus_name']
        aggregated_corpus_type = args['aggregated_corpus_type']
        
        # check valid aggregated_corpus_type
        if aggregated_corpus_type not in ['researcher', 'research_group']:
            return 'Invalid aggregated_corpus_type. It should be either "researcher" or "research_group".', 400
        
        try:
            sc.index_aggregated_corpus(aggregated_corpus_name, aggregated_corpus_type)
            return '', 200
        except Exception as e:
            return str(e), 500
        
delete_parser = reqparse.RequestParser()
delete_parser.add_argument(
    "aggregated_corpus_name", help="Name of the aggregated corpus to delete. For example, if the aggregated corpus we want to delete is stored in a file name 'aggregated_corpus.parquet', the aggregated_corpus_name should be 'aggregated_corpus' (without the extension quotes). Do not use quotes in the name of the aggregated corpus.")
@api.route('/deleteAggregatedCorpus/')
class deleteAggregatedCorpus(Resource):
    @api.doc(parser=delete_parser)
    def post(self):
        args = delete_parser.parse_args()
        aggregated_corpus_name = args['aggregated_corpus_name']
        
        try:
            sc.delete_aggregated_corpus(aggregated_corpus_name)
            return '', 200
        except Exception as e:
            return str(e), 500

@api.route('/listAggregatedCorpora/')
class listAggregatedCorpora(Resource):
    @api.doc()
    def get(self):
        try:
            return sc.list_ag_collections()
        except Exception as e:
            return str(e), 500