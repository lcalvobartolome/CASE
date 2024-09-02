package com.ewb;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.*;

import java.io.IOException;

/**
 * It provides a scorer that will iterate all documents in an index and compute
 * the JS similarity between a document and query document
 */
public class VectorSimQuery extends Query {
    String queryStr = "";
    Query q;

    public VectorSimQuery(Query subQuery) {
        this.q = subQuery;
    }

    public void setQueryString(String queryString) {
        this.queryStr = queryString;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, final ScoreMode scoreMode, float boost) throws IOException {
        Weight w;
        if (q == null) {
            w = new ConstantScoreWeight(this, boost) {
                public boolean isCacheable(LeafReaderContext leafReaderContext) {
                    return false;
                }

                @Override
                public Scorer scorer(LeafReaderContext context) throws IOException {
                    return new ConstantScoreScorer(this, score(), scoreMode,
                            DocIdSetIterator.all(context.reader().maxDoc()));
                }
            };
        } else {
            w = searcher.createWeight(q, scoreMode, boost);
        }
        return w;
    }

    @Override
    public String toString(String field) {
        return queryStr;
    }

    @Override
    public boolean equals(Object other) {
        return sameClassAs(other) &&
                queryStr.equals(other.toString());
    }

    @Override
    public int hashCode() {
        return classHash() ^ queryStr.hashCode();
    }

    @Override
    public void visit(QueryVisitor visitor) {
      visitor.visitLeaf(this);
    }

}