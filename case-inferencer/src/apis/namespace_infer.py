"""
This script defines a Flask RESTful namespace for performing inference.

Author: Lorena Calvo-Bartolomé
Date: 19/05/2023
"""

from flask_restx import Namespace, Resource, reqparse
from src.core.inferencer.case_inferencer import (CASECTMInferencer,
                                                CASEMalletInferencer,
                                                CASEProdLDAInferencer,
                                                CASESparkLDAInferencer)
from src.core.utils import get_infer_config
import logging
logging.basicConfig(level='DEBUG')
logger = logging.getLogger('Inferencer')

# ======================================================
# Define namespace for inference operations
# ======================================================
api = Namespace('Inference operations')

# Dictionary of available inferencers
inferencers = {
    "mallet": CASEMalletInferencer(api.logger),
    "sparkLDA": CASESparkLDAInferencer(api.logger),
    "prodLDA": CASEProdLDAInferencer(api.logger),
    "ctm": CASECTMInferencer(api.logger),
}

# ======================================================
# Define parsers to take inputs from user
# ======================================================
infer_doc_parser = reqparse.RequestParser()
infer_doc_parser.add_argument('text_to_infer',
                              help='Text to be inferred',
                              required=True)
infer_doc_parser.add_argument('model_for_infer',
                              help='Model to be used for the inference', required=True)

infer_corpus_parser = reqparse.RequestParser()


list_parser = reqparse.RequestParser()
list_parser.add_argument('argument',
                         help='To be defined',
                         required=True)

delete_parser = reqparse.RequestParser()
delete_parser.add_argument('argument',
                           help='To be defined',
                           required=True)


@api.route('/inferDoc/')
class InferDoc(Resource):
    @api.doc(parser=infer_doc_parser)
    def get(self):
        args = infer_doc_parser.parse_args()
        text_to_infer = args['text_to_infer']
        model_for_infer = args['model_for_infer']
        path_to_infer_config, trainer = \
            get_infer_config(logger=logger,
                             text_to_infer=text_to_infer,
                             model_for_infer=model_for_infer)
            
        try:
            return inferencers[trainer].predict(path_to_infer_config)
        except Exception as e:
            return str(e), 500
            
@api.route('/inferCorpus/')
class InferCorpus(Resource):
    @api.doc(parser=infer_corpus_parser)
    def get(self):
        # TODO
        pass


@api.route('/listInferenceModels/')
class listInferenceModels(Resource):
    @api.doc(parser=list_parser)
    def get(self):
        # TODO
        pass


@api.route('/deleteInferenceModel/')
class deleteInferenceModel(Resource):
    @api.doc(parser=delete_parser)
    def post(self):
        # TODO
        pass
