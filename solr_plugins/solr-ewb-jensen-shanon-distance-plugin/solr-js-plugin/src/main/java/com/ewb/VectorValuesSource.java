package com.ewb;

import org.apache.lucene.analysis.payloads.PayloadHelper;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.*;

/*
 * This class receives the query vector and computes its distance to the document vector by reading the vector values directly from the Lucene index. As distance metric, the Jensen-Shannon divergence is used.
 */
public class VectorValuesSource extends DoubleValuesSource {
    private final String field;
    // private final String metric;

    private Terms terms; // Access to the terms in a specific field
    private TermsEnum te; // Iterator to step through terms to obtain frequency information
    private String[] query_comps;

    public VectorValuesSource(String field, String strVector) {
        /*
         * Document queries are assumed to be given as:
         * http://localhost:8983/solr/{your-corpus-collection-name}/query?fl=name,score,
         * vector&q={!vp f=doctpc_{your-model-name} vector="t0|43 t4|548 t5|6 t20|403"}
         * while topic queries as follows:
         * http://localhost:8983/solr/{your-model-collection-name}/query?fl=name,score,
         * vector&q={!vp f=betas
         * vector="high|43 research|548 development|6 neural_networks|403"}
         */
        this.field = field;
        this.query_comps = strVector.split(" ");
    }

    public DoubleValues getValues(LeafReaderContext leafReaderContext, DoubleValues doubleValues) throws IOException {

        final LeafReader reader = leafReaderContext.reader();

        return new DoubleValues() {

            // Retrieves the payload value for each term in the document and calculates the
            // core based on vector lookup
            public double doubleValue() throws IOException {
                double score = 0;
                BytesRef text;
                String term = "";
                List<String> doc_topics = new ArrayList<String>();
                List<Integer> doc_probs = new ArrayList<Integer>();
                while ((text = te.next()) != null) {
                    term = text.utf8ToString();
                    if (term.isEmpty()) {
                        continue;
                    }

                    // Get the document probability distribution
                    float payloadValue = 0f;
                    PostingsEnum postings = te.postings(null, PostingsEnum.ALL);
                    // And after we get TermsEnum instance te, we can compute the document vector by
                    // iterating all payload components (we will have as many components as topics
                    // the model has)
                    while (postings.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                        int freq = postings.freq();
                        while (freq-- > 0)
                            postings.nextPosition();

                        BytesRef payload = postings.getPayload();
                        payloadValue = PayloadHelper.decodeInt(payload.bytes, payload.offset);
                        doc_topics.add(term);
                        // doc_topics.add(Integer.parseInt(term.substring(1)));
                        doc_probs.add((int) payloadValue);
                    }
                }

                // Create maps containing the value after '|' for each t that is present in both
                // strings for the case of document queries, and for each word that is present
                // in both strings for the case of topic queries
                Map<String, Integer> doc_values = new HashMap<>();
                Map<String, Integer> query_values = new HashMap<>();

                // Create pattern to match the document and topic queries
                Pattern pattern_docs = Pattern.compile("(t\\d+)\\|");
                Pattern pattern_words = Pattern.compile("([^|]+)\\|(\\d+)");

                for (String comp : query_comps) {
                    String key = "";
                    Matcher matcher;

                    matcher = pattern_docs.matcher(comp);
                    if (matcher.find()) {
                        key = matcher.group(1);
                    } else {
                        matcher = pattern_words.matcher(comp);
                        if (matcher.find()) {
                            key = matcher.group(1);
                        }
                    }

                    if (doc_topics.contains(key)) {
                        query_values.put(key, Integer.parseInt(comp.split("\\|")[1]));
                        doc_values.put(key, doc_probs.get(doc_topics.indexOf(key)));
                    }
                }

                // Convert the maps into arrays
                List<String> keys = new ArrayList<>(doc_values.keySet());

                double[] docProbabilities = new double[keys.size()];
                double[] queryProbabilities = new double[keys.size()];

                for (int i = 0; i < keys.size(); i++) {
                    String t = keys.get(i);
                    docProbabilities[i] = doc_values.get(t);
                    queryProbabilities[i] = query_values.get(t);
                }

                System.out.println(Arrays.toString(docProbabilities));
                System.out.println(Arrays.toString(queryProbabilities));

                Distance d = new Distance();

                // if (metric == "jensen-shannon") {
                // score = d.JensenShannonDivergence(docProbabilities, queryProbabilities);
                // }
                // else if (metric == "bhattacharyya") {
                // score = d.bhattacharyyaDistance(docProbabilities, queryProbabilities);
                // }

                score = d.bhattacharyyaDistance(docProbabilities, queryProbabilities);

                return score;

                // return score;
            }

            // Advance to next document (for each document in the LeafReaderContext)
            public boolean advanceExact(int doc) throws IOException {
                terms = reader.getTermVector(doc, field);
                if (terms == null) {
                    return false;
                }
                te = terms.iterator();
                return true;
            }
        };
    }

    public boolean needsScores() {
        return true;
    }

    public DoubleValuesSource rewrite(IndexSearcher indexSearcher) throws IOException {
        return this;
    }

    public int hashCode() {
        return 0;
    }

    public boolean equals(Object o) {
        return false;
    }

    public String toString() {
        return "JS(" + field + ",doc)";
    }

    public boolean isCacheable(LeafReaderContext leafReaderContext) {
        return false;
    }
}
