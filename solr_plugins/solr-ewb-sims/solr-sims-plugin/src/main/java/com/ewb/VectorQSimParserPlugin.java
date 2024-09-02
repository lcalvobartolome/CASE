package com.ewb;

import java.io.IOException;

import org.apache.lucene.queries.function.FunctionScoreQuery;
import org.apache.lucene.search.Query;
import org.apache.solr.common.SolrException;
import org.apache.solr.common.params.SolrParams;
import org.apache.solr.request.SolrQueryRequest;
import org.apache.solr.schema.FieldType;
import org.apache.solr.search.QParser;
import org.apache.solr.search.QParserPlugin;
import org.apache.solr.search.QueryParsing;
import org.apache.solr.search.SyntaxError;

public class VectorQSimParserPlugin extends QParserPlugin {
    @Override
    public QParser createParser(String qstr, SolrParams localParams, SolrParams params, SolrQueryRequest req) {
        return new QParser(qstr, localParams, params, req) {
            @Override
            public Query parse() throws SyntaxError {
                String field = localParams.get(QueryParsing.F);
                String vector = localParams.get("vector");

                if (field == null) {
                    throw new SolrException(SolrException.ErrorCode.BAD_REQUEST, "'f' no specified");
                }

                if (vector == null) {
                    throw new SolrException(SolrException.ErrorCode.BAD_REQUEST, "vector missing");
                }

                String[] limits = vector.split(",");

                double low_limit, up_limit;
               
                low_limit = Double.parseDouble(limits[0]);
                up_limit = Double.parseDouble(limits[1]);
                
                if (low_limit < 0 || low_limit > 100 || up_limit < 0 || up_limit > 100) {
                    throw new SolrException(SolrException.ErrorCode.BAD_REQUEST, "Limits must be between 0 and 100%.");
                }

                if (low_limit >= up_limit) {
                    throw new SolrException(SolrException.ErrorCode.BAD_REQUEST, "The lower limit must be lower than the upper limit");
                }
            
                Query subQuery = subQuery(localParams.get(QueryParsing.V), null).getQuery();

                FieldType ft = req.getCore().getLatestSchema().getFieldType(field);
                if (ft != null) {
                    VectorSimQuery q = new VectorSimQuery(subQuery);
                    q.setQueryString(localParams.toLocalParamsString());
                    query = q;
                }

                if (query == null) {
                    throw new SolrException(SolrException.ErrorCode.BAD_REQUEST, "Query is null");
                }

                return new FunctionScoreQuery(query, new VectorSimValuesSource(field, vector));
            }
        };
    }

    @Override
    public void close() throws IOException {
        // Do nothing
    }
}
