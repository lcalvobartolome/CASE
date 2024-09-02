package com.ewb;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

public class DistanceTest {
    @Test
    public void testJensenShannonDivergence1() {
        System.out.println("Starting test 1...");
        double[] p = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        double[] q = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        double score = 0;
        Distance d = new Distance();
        score = d.JensenShannonDivergence(p, q);

        // assertTrue(MathEx.KullbackLeiblerDivergence(prob, p) < 0.05);
        System.out.println(score);

    }

    @Test
    public void testJensenShannonDivergence2() {
        System.out.println("Starting test 2...");
        double[] p = { 1, 2, 3, 4, 5, 6, 7 };
        double[] q = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        double score = 0;
        Distance d = new Distance();
        try {
            score = d.JensenShannonDivergence(p, q);
            System.out.println(score);
        } catch (Exception e) {
            System.out.println(e);
        }
    }

    @Test
    public void testJensenShannonDivergence3() {
        System.out.println("Starting test 3...");
        double[] p = { 0, 105, 0, 0, 0, 0, 471, 0, 15, 0, 0, 120, 0, 0, 71, 0, 0, 0, 0, 0, 218, 0, 0, 0, 0 };
        double[] q = { 0, 4, 0, 1, 0, 4, 0, 4, 0, 1, 0, 4, 0, 4, 0, 1, 0, 4, 5, 3, 4, 3, 2, 0, 0 };

        double score = 0;
        Distance d = new Distance();
        score = d.JensenShannonDivergence(p, q);

        // assertTrue(MathEx.KullbackLeiblerDivergence(prob, p) < 0.05);
        System.out.println(score);

    }

    @Test
    public void testGetInteresction() {
        System.out.println("Starting test 4...");
        String query_vector = "t0|38 t1|840 t6|122";
        String doc_vector = "t0|43 t4|548 t5|6 t20|403";

        String[] query_comps = query_vector.split(" ");

        List<Integer> doc_topics = new ArrayList<Integer>();
        List<Integer> doc_probs = new ArrayList<Integer>();

        for (String comp : doc_vector.split(" ")) {
            int tpc_id = Integer.parseInt(comp.split("\\|")[0].split("t")[1]);
            doc_topics.add(tpc_id);
            doc_probs.add(Integer.parseInt(comp.split("\\|")[1]));
        }
        System.out.println(doc_topics);
        System.out.println(doc_probs);

        Map<Integer, Integer> doc_values = new HashMap<>();
        Map<Integer, Integer> query_values = new HashMap<>();

        for (String comp : query_comps) {
            int tpc_id = Integer.parseInt(comp.split("\\|")[0].split("t")[1]);
            System.out.println("tpc_id: " + tpc_id);
            if (doc_topics.contains(tpc_id)) {
                query_values.put(tpc_id, Integer.parseInt(comp.split("\\|")[1]));
                doc_values.put(tpc_id, doc_probs.get(doc_topics.indexOf(tpc_id)));
            }
        }

        // Convert the maps into arrays
        List<Integer> sortedKeys = new ArrayList<>(doc_values.keySet());
        Collections.sort(sortedKeys);

        double[] docProbabilities = new double[sortedKeys.size()];
        double[] queryProbabilities = new double[sortedKeys.size()];

        for (int i = 0; i < sortedKeys.size(); i++) {
            Integer t = sortedKeys.get(i);
            docProbabilities[i] = doc_values.get(t);
            queryProbabilities[i] = query_values.get(t);
        }

        System.out.println(Arrays.toString(docProbabilities));
        System.out.println(Arrays.toString(queryProbabilities));

        double score = 0;
        Distance d = new Distance();
        score = d.bhattacharyyaDistance(docProbabilities, queryProbabilities);

        System.out.println("bhattacharyya: " + score);

        score = d.JensenShannonDivergence(docProbabilities, queryProbabilities);

        System.out.println("jensen: " + score);
    }
}
