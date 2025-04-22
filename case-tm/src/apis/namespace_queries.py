"""
This script defines a Flask RESTful namespace for managing Solr queries.

Author: Lorena Calvo-Bartolomé
Date: 13/04/2023
"""

import json
import pathlib
from flask_restx import Namespace, Resource, reqparse
from src.core.clients.case_solr_client import CASESolrClient

# ======================================================
# Define namespace for managing queries
# ======================================================
api = Namespace(
    'Queries', description='Specfic Solr queries.')

# ======================================================
# Namespace variables
# ======================================================
# Create Solr client
sc = CASESolrClient(api.logger)

# Define parsers to take inputs from user
q1_parser = reqparse.RequestParser()
q1_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q1_parser.add_argument(
    'doc_id', help='ID of the document whose doc-topic distribution associated to a specific model is to be retrieved', required=True)
q1_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution to be retrieved', required=True)

q2_parser = reqparse.RequestParser()
q2_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)

q3_parser = reqparse.RequestParser()
q3_parser.add_argument(
    'collection', help='Name of the collection', required=True)

q4_parser = reqparse.RequestParser()
q4_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q4_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
q4_parser.add_argument(
    'topic_id', help='Topic whose proportion must be larger than a certain threshold', required=True)
q4_parser.add_argument(
    'threshold', help='Query threshold', required=True)
q4_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content.', required=False)
q4_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q5_parser = reqparse.RequestParser()
q5_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q5_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
q5_parser.add_argument(
    'doc_id', help="ID of the document whose similarity is going to be checked against all other documents in 'corpus_collection'", required=True)
q5_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q5_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q6_parser = reqparse.RequestParser()
q6_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q6_parser.add_argument(
    'doc_id', help="ID of the document whose metadata is going to be retrieved'", required=True)

q7_parser = reqparse.RequestParser()
q7_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q7_parser.add_argument(
    'string', help="String to be search in the SearcheableField field'", required=True)
q7_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q7_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q8_parser = reqparse.RequestParser()
q8_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q8_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q8_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q9_parser = reqparse.RequestParser()
q9_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q9_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
q9_parser.add_argument(
    'topic_id', help="ID of the topic whose top documents according to 'model_name' are being searched", required=True)
q9_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q9_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q10_parser = reqparse.RequestParser()
q10_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q10_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q10_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q11_parser = reqparse.RequestParser()
q11_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q11_parser.add_argument(
    'topic_id', help='ID of the topic whose whose word-topic distribution is to be retrieved', required=False)

q12_parser = reqparse.RequestParser()
q12_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q12_parser.add_argument(
    'topic_id', help='ID of the topic whose whose word-topic distribution is to be retrieved', required=False)
q12_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q12_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q13_parser = reqparse.RequestParser()
q13_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q13_parser.add_argument(
    'model_name', help='Name of the model collection', required=True)
q13_parser.add_argument(
    'lower_limit', help='Lower percentage of semantic similarity to retrieve pairs of documents', required=True)
q13_parser.add_argument(
    'upper_limit', help='Upper percentage of semantic similarity to retrieve pairs of documents', required=True)
q13_parser.add_argument(
    'year', help='Publication year to be filtered by', required=False)
q13_parser.add_argument(
    'num_records', help='Controls how many pairs of documents are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q14_parser = reqparse.RequestParser()
q14_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q14_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
q14_parser.add_argument(
    'text_to_infer', help="Text to be inferred", required=True)
q14_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q14_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q15_parser = reqparse.RequestParser()
q15_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q15_parser.add_argument(
    'doc_id', help='ID of the document whose whose doc-topic distribution associated to a specific model is to be retrieved', required=True)

q16_parser = reqparse.RequestParser()
q16_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
q16_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution to be retrieved', required=True)
q16_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q16_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q17_parser = reqparse.RequestParser()
q17_parser.add_argument(
    'model_name', help='Name of the model to retrieve the topic-word distribution.', required=True)
q17_parser.add_argument(
    'tpc_id', help='ID of the specific topic to retrieve the topic-word distribution.', required=True)
q17_parser.add_argument(
    'word', help='Word of interest to retrieve its topic-word distribution in the specified topic. If the word is not present, the distribution is 0.', required=True)

q18_parser = reqparse.RequestParser()
q18_parser.add_argument(
    'corpus_name', help='Name of the corpus.', required=True)
q18_parser.add_argument(
    'ids', help='IDs of the documents separated by commas', required=True)
q18_parser.add_argument(
    'words', help='Words to get the BOW from, separated by commas', required=True)
q18_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q18_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

q19_parser = reqparse.RequestParser()
q19_parser.add_argument(
    'model_collection', help='Name of the model collection', required=True)
q19_parser.add_argument(
    'user', help='Identifier/name of the user whose relevant topics are going to be retrieved', required=True)
q19_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
q19_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)



@api.route('/getThetasDocById/')
class getThetasDocById(Resource):
    @api.doc(parser=q1_parser)
    def get(self):
        args = q1_parser.parse_args()
        corpus_collection = args['corpus_collection']
        doc_id = args['doc_id']
        model_name = args['model_name']

        try:
            return sc.do_Q1(corpus_col=corpus_collection,
                            doc_id=doc_id,
                            model_name=model_name)
        except Exception as e:
            return str(e), 500


@api.route('/getCorpusMetadataFields/')
class getCorpusMetadataFields(Resource):
    @api.doc(parser=q2_parser)
    def get(self):
        args = q2_parser.parse_args()
        corpus_collection = args['corpus_collection']
        try:
            return sc.do_Q2(corpus_col=corpus_collection)
        except Exception as e:
            return str(e), 500


@api.route('/getNrDocsColl/')
class getNrDocsColl(Resource):
    @api.doc(parser=q3_parser)
    def get(self):
        args = q3_parser.parse_args()
        collection = args['collection']
        try:
            return sc.do_Q3(col=collection)
        except Exception as e:
            return str(e), 500


@api.route('/getDocsWithThetasLargerThanThr/')
class getDocsWithThetasLargerThanThr(Resource):
    @api.doc(parser=q4_parser)
    def get(self):
        args = q4_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        topic_id = args['topic_id']
        threshold = args['threshold']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q4(corpus_col=corpus_collection,
                            model_name=model_name,
                            topic_id=topic_id,
                            thr=threshold,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getDocsWithHighSimWithDocByid/')
class getDocsWithHighSimWithDocByid(Resource):
    @api.doc(parser=q5_parser)
    def get(self):
        args = q5_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        doc_id = args['doc_id']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q5(corpus_col=corpus_collection,
                            model_name=model_name,
                            doc_id=doc_id,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getMetadataDocById/')
class getMetadataDocById(Resource):
    @api.doc(parser=q6_parser)
    def get(self):
        args = q6_parser.parse_args()
        corpus_collection = args['corpus_collection']
        doc_id = args['doc_id']

        try:
            return sc.do_Q6(corpus_col=corpus_collection,
                            doc_id=doc_id)
        except Exception as e:
            return str(e), 500


@api.route('/getDocsWithString/')
class getDocsWithString(Resource):
    @api.doc(parser=q7_parser)
    def get(self):
        args = q7_parser.parse_args()
        corpus_collection = args['corpus_collection']
        string = args['string']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q7(corpus_col=corpus_collection,
                            string=string,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getTopicsLabels/')
class getTopicsLabels(Resource):
    @api.doc(parser=q8_parser)
    def get(self):
        args = q8_parser.parse_args()
        model_collection = args['model_collection']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q8(model_col=model_collection,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getTopicTopDocs/')
class getTopicTopDocs(Resource):
    @api.doc(parser=q9_parser)
    def get(self):
        args = q9_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        topic_id = args['topic_id']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q9(corpus_col=corpus_collection,
                            model_name=model_name,
                            topic_id=topic_id,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500

@api.route('/getModelInfo/')
class getModelInfo(Resource):
    @api.doc(parser=q10_parser)
    def get(self):
        args = q10_parser.parse_args()
        model_collection = args['model_collection']
        start = args['start']
        rows = args['rows']
        try:
            return sc.do_Q10(model_col=model_collection,
                            start=start,
                            rows=rows,
                            only_id=False)
        except Exception as e:
            return str(e), 500


@api.route('/getBetasTopicById/')
class getBetasTopicById(Resource):
    @api.doc(parser=q11_parser)
    def get(self):
        args = q11_parser.parse_args()
        model_collection = args['model_collection']
        topic_id = args['topic_id']
        
        try:
            return sc.do_Q11(model_col=model_collection,
                            topic_id=topic_id)
        except Exception as e:
            return str(e), 500


@api.route('/getMostCorrelatedTopics/')
class getMostCorrelatedTopics(Resource):
    @api.doc(parser=q12_parser)
    def get(self):
        args = q12_parser.parse_args()
        model_collection = args['model_collection']
        topic_id = args['topic_id']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q12(model_col=model_collection,
                            topic_id=topic_id,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500

@api.route('/getPairsOfDocsWithHighSim/')
class getPairsOfDocsWithHighSim(Resource):
    @api.doc(parser=q13_parser)
    def get(self):
        args = q13_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        lower_limit = args['lower_limit']
        upper_limit = args['upper_limit']
        year = args['year']
        num_records = args['num_records']

        try:
            return sc.do_Q13(corpus_col=corpus_collection,
                            model_name=model_name,
                            lower_limit=lower_limit,
                            upper_limit=upper_limit,
                            year=year,
                            num_records=num_records)
        except Exception as e:
            return str(e), 500

@api.route('/getDocsSimilarToFreeText/')
class getDocsSimilarToFreeText(Resource):
    @api.doc(parser=q14_parser)
    def get(self):
        args = q14_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        text_to_infer = args['text_to_infer']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q14(corpus_col=corpus_collection,
                            model_name=model_name,
                            text_to_infer=text_to_infer,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500


@api.route('/getLemmasDocById/')
class getLemmasDocById(Resource):
    @api.doc(parser=q15_parser)
    def get(self):
        args = q15_parser.parse_args()
        corpus_collection = args['corpus_collection']
        doc_id = args['doc_id']

        try:
            return sc.do_Q15(corpus_col=corpus_collection,
                            doc_id=doc_id)
        except Exception as e:
            return str(e), 500


@api.route('/getThetasAndDateAllDocs/')
class getThetasAndDateAllDocs(Resource):
    @api.doc(parser=q16_parser)
    def get(self):
        args = q16_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q16(corpus_col=corpus_collection,
                            model_name=model_name,
                            start=start,
                            rows=rows)
        except Exception as e:
            return str(e), 500

@api.route('/getBetasByWordAndTopicId/')
class getBetasByWordAndTopicId(Resource):
    @api.doc(parser=q17_parser)
    def get(self):
        args = q17_parser.parse_args()
        model_name = args['model_name']
        tpc_id = args['tpc_id']
        word = args['word']

        try:
            return sc.do_Q17(model_name=model_name,
                            tpc_id=tpc_id,
                            word=word)
        except Exception as e:
            return str(e), 500
        
@api.route('/getBOWbyDocsIDs/')
class getBOWbyDocsIDs(Resource):
    @api.doc(parser=q18_parser)
    def get(self):
        args = q18_parser.parse_args()
        corpus_name = args['corpus_name']
        ids = args['ids']
        words = args['words']
        start = args['start']
        rows = args['rows']

        try:
            return sc.do_Q18(corpus_col=corpus_name,
                            ids=ids.split(","),
                            words=words,
                            start=start,
                            rows=rows,)
        except Exception as e:
            return str(e), 500
        
@api.route('/getUserRelevantTopics/')
class getUserRelevantTopics(Resource):
    @api.doc(parser=q19_parser)
    def get(self):
        args = q19_parser.parse_args()
        model_collection = args['model_collection']
        user = args['user']
        start = args['start']
        rows = args['rows']
        try:
            return sc.do_Q19(model_col=model_collection,
                            start=start,
                            rows=rows,
                            user=user)
        except Exception as e:
            return str(e), 500
        
# ======================================================
# RECOMMENDER SYSTEM QUERIES
# ======================================================

# ------------------------------------------------------
# getTopicStatistics
# ------------------------------------------------------
# corpus_collection, model_name, topic_id
getTopicStatistics_parser = reqparse.RequestParser()
getTopicStatistics_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
getTopicStatistics_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=True)
getTopicStatistics_parser.add_argument(
    'topic_id', help='ID of the topic whose statistics are being searched', required=True)
@api.route('/getTopicStatistics/')
class getTopicStatistics(Resource):
    @api.doc(parser=getTopicStatistics_parser)
    def get(self):
        args = getTopicStatistics_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        topic_id = args['topic_id']

        try:
            with open("/case-tm/src/apis/dummies/getTopicStatistics.json", "r") as file:
                data = json.load(file)
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 500

# ------------------------------------------------------
# getTopicTopResearchers
# ------------------------------------------------------
# corpus_collection, model_name, topic_id, start, rows
getTopicTopResearchers_parser = reqparse.RequestParser()
getTopicTopResearchers_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=False)
getTopicTopResearchers_parser.add_argument(
    'model_name', help='Name of the model responsible for the creation of the doc-topic distribution', required=True)
getTopicTopResearchers_parser.add_argument(
    'topic_id', help="ID of the topic whose top researchers are being searched", required=True)
getTopicTopResearchers_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
getTopicTopResearchers_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

@api.route('/getTopicTopResearchers/')
class getTopicTopResearchers(Resource):
    @api.doc(parser=getTopicTopResearchers_parser)
    def get(self):
        args = getTopicTopResearchers_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        topic_id = args['topic_id']
        start = args['start']
        rows = args['rows']
        
        try:
            # @TODO: Implement this query
            #with open("/case-tm/src/apis/dummies/getTopicTopResearchers.json", "r") as file:
            #    data = json.load(file)
            #return data, 200
            return sc.do_Q20(agg_corpus_col="uc3m_researchers",#corpus_collection,
                            model_name=model_name,
                            topic_id=topic_id,
                            start=start,
                            rows=rows)
        except Exception as e:
            return {"error": str(e)}, 500
        
# ------------------------------------------------------
# getTopicTopRGs
# ------------------------------------------------------
# similar tp getTopicTopResearchers, but for Research Groups
getTopicTopRGs_parser = reqparse.RequestParser()
getTopicTopRGs_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=False)
getTopicTopRGs_parser.add_argument(
    'model_name', help='Name of the model responsible for the creation of the doc-topic distribution', required=True)
getTopicTopRGs_parser.add_argument(
    'topic_id', help="ID of the topic whose top research groups are being searched", required=True)
getTopicTopRGs_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
getTopicTopRGs_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

@api.route('/getTopicTopRGs/')
class getTopicTopRGs(Resource):
    @api.doc(parser=getTopicTopRGs_parser)
    def get(self):
        args = getTopicTopRGs_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        topic_id = args['topic_id']
        start = args['start']
        rows = args['rows']
        
        try:
            ## @TODO: Implement this query
            #with open("/case-tm/src/apis/dummies/getTopicTopRGs.json", "r") as file:
            #    data = json.load(file)
            #return data, 200
            return sc.do_Q20(agg_corpus_col="uc3m_research_groups",#corpus_collection,
                            model_name=model_name,
                            topic_id=topic_id,
                            start=start,
                            rows=rows)
        except Exception as e:
            return {"error": str(e)}, 500

# ------------------------------------------------------
# getMetadataAGByID
# ------------------------------------------------------
getMetadataAGByID_parser = reqparse.RequestParser()
getMetadataAGByID_parser.add_argument(
    'aggregated_collection_name', help='Name of the aggregated collection (uc3m_researchers or uc3m_research_groups)', required=True)
getMetadataAGByID_parser.add_argument(
    'id', help='ID of the aggregated collection', required=True)
@api.route('/getMetadataAGByID/')
class getMetadataAGByID(Resource):
    @api.doc(parser=getMetadataAGByID_parser)
    def get(self):
        args = getMetadataAGByID_parser.parse_args()
        ag_collection = args['aggregated_collection_name']
        id = args['id']
        
        # one of the two must be provided
        # if not ag_collection and not id:
        #    return {"error": "One of the two parameters must be provided"}, 400
        
        # file_r = f"/case-tm/src/apis/dummies/getMetadataAGByID_r_{id}.json"
        # file_rg = f"/case-tm/src/apis/dummies/getMetadataAGByID_rg_{id}.json"
        
        # one of the two files must exist
        #if ag_collection.lower() == "uc3m_researchers":
        #    file = file_r
        #elif ag_collection.lower() == "uc3m_research_groups":
        #    file = file_rg
        try:
            #with open(file, "r") as file:
            #    data = json.load(file)
            return sc.do_Q21(agg_corpus_col=ag_collection,
                            doc_id=id)
        except Exception as e:
            return {"error": str(e)}, 500

# ------------------------------------------------------
# getTopicEvolution
# ------------------------------------------------------
# corpus_collection, model_name, topic_id
getTopicEvolution_parser = reqparse.RequestParser()
getTopicEvolution_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
getTopicEvolution_parser.add_argument(
    'model_name', help='Name of the model responsible for the creation of the doc-topic distribution', required=True)
getTopicEvolution_parser.add_argument(
    'topic_id', help='ID of the topic whose evolution is being searched', required=True)
getTopicEvolution_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
getTopicEvolution_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

@api.route('/getTopicEvolution/')
class getTopicEvolution(Resource):
    @api.doc(parser=getTopicEvolution_parser)
    def get(self):
        args = getTopicEvolution_parser.parse_args()
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        topic_id = args['topic_id']
        start = args['start']
        rows = args['rows']
        
        try:
            # @TODO: Implement this query
            with open("/case-tm/src/apis/dummies/getTopicEvolution.json", "r") as file:
                data = json.load(file)
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 500
        
# ------------------------------------------------------
# getAGDocsWithString
# ------------------------------------------------------
getAGDocsWithString_parser = reqparse.RequestParser()
getAGDocsWithString_parser.add_argument(
    'aggregated_collection_name', help='Name of the aggregated collection', required=False)
getAGDocsWithString_parser.add_argument(
    'string', help='String to be search in the SearcheableField field', required=False)
getAGDocsWithString_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
getAGDocsWithString_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

@api.route('/getAGDocsWithString/')
class getAGDocsWithString(Resource):
    @api.doc(parser=getAGDocsWithString_parser)
    def get(self):
        args = getAGDocsWithString_parser.parse_args()
        ag_collection = args['aggregated_collection_name']
        string = args['string']
        start = args['start']
        rows = args['rows']
        
        try:
            # @TODO: Implement this query
            with open("/case-tm/src/apis/dummies/getAGDocsWithString.json", "r") as file:
                data = json.load(file)
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 500

# ------------------------------------------------------
# getSimiliarityCriteriaList
# ------------------------------------------------------
@api.route('/getSimiliarityCriteriaList/')
class getSimiliarityCriteriaList(Resource):
    def get(self):
        try:
            # @TODO: Implement this query
            with open("/case-tm/src/apis/dummies/getSimiliarityCriteriaList.json", "r") as file:
                data = json.load(file)
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 500

# ------------------------------------------------------
# getResearchersSimilarToCall
# ------------------------------------------------------
# IdCall, SimilarityMethod, [filtering options, ranking options]
# IdCall comes from FundingCallsHE --> id
getResearchersSimilarToCall_parser = reqparse.RequestParser()
getResearchersSimilarToCall_parser.add_argument(
    'id', help='ID of the call', required=True)
getResearchersSimilarToCall_parser.add_argument(
    'similarity_method', help='Similarity method to be used', required=True)
getResearchersSimilarToCall_parser.add_argument(
    'filtering_options', help='Filtering options', required=False)
getResearchersSimilarToCall_parser.add_argument(
    'ranking_options', help='Ranking options', required=False)
getResearchersSimilarToCall_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
getResearchersSimilarToCall_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

@api.route('/getResearchersSimilarToCall/')
class getResearchersSimilarToCall(Resource):
    @api.doc(parser=getResearchersSimilarToCall_parser)
    def get(self):
        args = getResearchersSimilarToCall_parser.parse_args()
        id = args['id']
        similarity_method = args['similarity_method']
        filtering_options = args['filtering_options']
        ranking_options = args['ranking_options']
        start = args['start']
        rows = args['rows']
        
        try:
            # @TODO: Implement this query
            # check that similarity_method is valid
            with open("/case-tm/src/apis/dummies/getResearchersSimilarToCall.json", "r") as file:
                data = json.load(file)
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 500

# ------------------------------------------------------
# getResearchGroupsSimilarToCall
# ------------------------------------------------------
getResearchGroupsSimilarToCall_parser = reqparse.RequestParser()
getResearchGroupsSimilarToCall_parser.add_argument(
    'id', help='ID of the call', required=True)
getResearchGroupsSimilarToCall_parser.add_argument(
    'similarity_method', help='Similarity method to be used', required=True)
getResearchGroupsSimilarToCall_parser.add_argument(
    'filtering_options', help='Filtering options', required=False)
getResearchGroupsSimilarToCall_parser.add_argument(
    'ranking_options', help='Ranking options', required=False)
getResearchGroupsSimilarToCall_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
getResearchGroupsSimilarToCall_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

@api.route('/getResearchGroupsSimilarToCall/')
class getResearchGroupsSimilarToCall(Resource):
    @api.doc(parser=getResearchGroupsSimilarToCall_parser)
    def get(self):
        args = getResearchGroupsSimilarToCall_parser.parse_args()
        id = args['id']
        similarity_method = args['similarity_method']
        filtering_options = args['filtering_options']
        ranking_options = args['ranking_options']
        start = args['start']
        rows = args['rows']
        
        try:
            # @TODO: Implement this query
            # check that similarity_method is valid
            with open("/case-tm/src/apis/dummies/getResearchGroupsSimilarToCall.json", "r") as file:
                data = json.load(file)
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 500

# ------------------------------------------------------
# getResearchersSimilarToText
# ------------------------------------------------------
getResearchersSimilarToText_parser = reqparse.RequestParser()
getResearchersSimilarToText_parser.add_argument(
    'text', help='Text to be inferred', required=True)
getResearchersSimilarToText_parser.add_argument(
    'similarity_method', help='Similarity method to be used', required=True)
getResearchersSimilarToText_parser.add_argument(
    'filtering_options', help='Filtering options', required=False)
getResearchersSimilarToText_parser.add_argument(
    'ranking_options', help='Ranking options', required=False)
getResearchersSimilarToText_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
getResearchersSimilarToText_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

@api.route('/getResearchersSimilarToText/')
class getResearchersSimilarToText(Resource):
    @api.doc(parser=getResearchersSimilarToText_parser)
    def get(self):
        args = getResearchersSimilarToText_parser.parse_args()
        text = args['text']
        similarity_method = args['similarity_method']
        filtering_options = args['filtering_options']
        ranking_options = args['ranking_options']
        start = args['start']
        rows = args['rows']
        
        try:
            # @TODO: Implement this query
            # check that similarity_method is valid
            with open("/case-tm/src/apis/dummies/getResearchersSimilarToText.json", "r") as file:
                data = json.load(file)
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 500

# ------------------------------------------------------
# getResearchGroupsSimilarToText
# ------------------------------------------------------
getResearchGroupsSimilarToText_parser = reqparse.RequestParser()
getResearchGroupsSimilarToText_parser.add_argument(
    'text', help='Text to be inferred', required=True)
getResearchGroupsSimilarToText_parser.add_argument(
    'similarity_method', help='Similarity method to be used', required=True)
getResearchGroupsSimilarToText_parser.add_argument(
    'filtering_options', help='Filtering options', required=False)
getResearchGroupsSimilarToText_parser.add_argument(
    'ranking_options', help='Ranking options', required=False)
getResearchGroupsSimilarToText_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
getResearchGroupsSimilarToText_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

@api.route('/getResearchGroupsSimilarToText/')
class getResearchGroupsSimilarToText(Resource):
    @api.doc(parser=getResearchGroupsSimilarToText_parser)
    def get(self):
        args = getResearchGroupsSimilarToText_parser.parse_args()
        text = args['text']
        similarity_method = args['similarity_method']
        filtering_options = args['filtering_options']
        ranking_options = args['ranking_options']
        start = args['start']
        rows = args['rows']
        
        try:
            # @TODO: Implement this query
            # check that similarity_method is valid
            with open("/case-tm/src/apis/dummies/getResearchGroupsSimilarToText.json", "r") as file:
                data = json.load(file)
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 500

# ------------------------------------------------------
# getCallsSimilarToResearcher
# ------------------------------------------------------
# invID, SimilarityMethod, collection_name (funding_calls)
getCallsSimilarToResearcher_parser = reqparse.RequestParser()
getCallsSimilarToResearcher_parser.add_argument(
    'id', help='ID of the researcher', required=True)
getCallsSimilarToResearcher_parser.add_argument(
    'similarity_method', help='Similarity method to be used', required=True)
getCallsSimilarToResearcher_parser.add_argument(
    'collection_name', help='Name of the collection', required=True)
getCallsSimilarToResearcher_parser.add_argument(
    'filtering_options', help='Filtering options', required=False)
getCallsSimilarToResearcher_parser.add_argument(
    'ranking_options', help='Ranking options', required=False)
getCallsSimilarToResearcher_parser.add_argument(
    'start', help='Specifies an offset (by default, 0) into the responses at which Solr should begin displaying content', required=False)
getCallsSimilarToResearcher_parser.add_argument(
    'rows', help='Controls how many rows of responses are displayed at a time (default value: maximum number of docs in the collection)', required=False)

@api.route('/getCallsSimilarToResearcher/')
class getCallsSimilarToResearcher(Resource):
    @api.doc(parser=getCallsSimilarToResearcher_parser)
    def get(self):
        args = getCallsSimilarToResearcher_parser.parse_args()
        id = args['id']
        similarity_method = args['similarity_method']
        collection_name = args['collection_name']
        filtering_options = args['filtering_options']
        ranking_options = args['ranking_options']
        start = args['start']
        rows = args['rows']
        
        try:
            # @TODO: Implement this query
            # check that similarity_method is valid
            with open("/case-tm/src/apis/dummies/getCallsSimilarToResearcher.json", "r") as file:
                data = json.load(file)
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 500

# ------------------------------------------------------
# getThetasResearcherByID
# ------------------------------------------------------
# this needs to be called twice, once with papers, once with cordis.
# model_name can be left empty (default model for the collection will be used)
# invID, corpus_collection, model_name
getThetasResearcherByID_parser = reqparse.RequestParser()
getThetasResearcherByID_parser.add_argument(
    'id', help='ID of the researcher', required=True)
getThetasResearcherByID_parser.add_argument(
    'corpus_collection', help='Name of the corpus collection', required=True)
getThetasResearcherByID_parser.add_argument(
    'model_name', help='Name of the model reponsible for the creation of the doc-topic distribution', required=False)

@api.route('/getThetasResearcherByID/')
class getThetasResearcherByID(Resource):
    @api.doc(parser=getThetasResearcherByID_parser)
    def get(self):
        args = getThetasResearcherByID_parser.parse_args()
        id = args['id']
        corpus_collection = args['corpus_collection']
        model_name = args['model_name']
        
        try:
            # @TODO: Implement this query
            with open("/case-tm/src/apis/dummies/getThetasResearcherByID.json", "r") as file:
                data = json.load(file)
            return data, 200
        except Exception as e:
            return {"error": str(e)}, 500
