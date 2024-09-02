package com.ewb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
    public void testFirstApproach() {
        System.out.println("Starting test 4...");
        String query_vector = "20,50";
        String doc_vector = "218417|0.68 831809|0.33 314692|0.43 717081|0.12";

        String[] limits = query_vector.split(",");
    
        List<String> doc_id = new ArrayList<String>();
        List<Double> doc_sim = new ArrayList<Double>();

        for (String comp : doc_vector.split(" ")) {
            String tpc_id = comp.split("\\|")[0];
            doc_id.add(tpc_id);
            doc_sim.add(Double.parseDouble(comp.split("\\|")[1]));
        }
        System.out.println(doc_id);
        System.out.println(doc_sim);

        double lowerLimit = Double.parseDouble(limits[0])/100;
        double upperLimit = Double.parseDouble(limits[1])/100;

        System.out.println(lowerLimit);
        System.out.println(upperLimit);

        // Step 1: Filter the docSimilarity within the lower and upper limits
        List<docSimilarity> filteredSimilarities = new ArrayList<>();
        for (int i = 0; i < doc_id.size(); i++) {
            String id = doc_id.get(i);
            double similarity = doc_sim.get(i);

            if (similarity >= lowerLimit && similarity <= upperLimit) {
                filteredSimilarities.add(new docSimilarity(id, similarity));
            }
        }
        for (docSimilarity d : filteredSimilarities) {
            System.out.println("Id: "+ d.getId()+ ", Similarity: " +  d.getSimilarity());
        }
        System.out.println("Now, we order the list");
        // Step 2: Order filtered similarities in descendent order
        Collections.sort(filteredSimilarities, new SimilarityComparator().reversed());

        for (docSimilarity d : filteredSimilarities) {
            System.out.println("Id: "+ d.getId()+ ", Similarity: " +  d.getSimilarity());
        }

        System.out.println(filteredSimilarities);
    }

    @Test
    public void testSecondApproach() {
        System.out.println("Starting test 5...");
        double score = 0.0;
        String query_vector = "0,99.99";
        //String doc_vector = "314692|0.43 218417|0.68 717081|0.12 831809|0.33 831809|0.26 831809|0.50";
        String doc_vector = "692925|0.9999999403953552 715747|0.9999998211860657 740900|0.9999996423721313 306260|0.9999996423721313 268343|0.9999988675117493 329098|0.9999978542327881 239606|0.9999978542327881 804176|0.9999948143959045 682922|0.9999929070472717 646649|0.9999920725822449 702139|0.9999918341636658 702971|0.9999905824661255 844629|0.999988317489624 234847|0.9999812841415405 681908|0.9999797344207764 224770|0.9999596476554871 228046|0.9999579191207886 321933|0.9999557733535767 203134|0.9999427199363708 101030083|0.9999029040336609 293975|0.9999029040336609 757646|0.9996441006660461 724939|0.9996145963668823 682150|0.9996052980422974 715734|0.9995374083518982 949499|0.9995003342628479 101002188|0.9987278580665588 306493|0.9983030557632446 757802|0.9982131719589233 627195|0.9981504082679749 334515|0.9980514049530029 757535|0.9976450204849243 792862|0.9969668984413147 239605|0.9968035817146301 615722|0.996498167514801 320389|0.9964777231216431 251871|0.9964262843132019 770127|0.9962782263755798 772466|0.9962424039840698 690904|0.9962361454963684 628104|0.9962361454963684 677120|0.9962361454963684 802107|0.9962361454963684 321749|0.9962361454963684 306614|0.9962361454963684 274345|0.9962361454963684 945655|0.9962361454963684 307663|0.9962361454963684 804208|0.9962361454963684";

        String[] limits = query_vector.split(",");
    
        List<String> doc_id = new ArrayList<String>();
        List<Double> doc_sim = new ArrayList<Double>();

        for (String comp : doc_vector.split(" ")) {
            String tpc_id = comp.split("\\|")[0];
            doc_id.add(tpc_id);
            doc_sim.add(Double.parseDouble(comp.split("\\|")[1]));
        }

        System.out.println(doc_sim);

        double lowerLimit = Double.parseDouble(limits[0])/100;
        double upperLimit = Double.parseDouble(limits[1])/100;
        
        System.out.println(lowerLimit);
        System.out.println(upperLimit);

        int low_index = -1;
        int up_index = -1;

        // Combine lists into a list of docSimilarity objects
        List<docSimilarity> docSimilarities = new ArrayList<>();
        for (int i = 0; i < doc_id.size(); i++) {
            docSimilarities.add(new docSimilarity(doc_id.get(i), doc_sim.get(i)));
        }

        // Sort the docSimilarities list using SimilarityComparator
        Collections.sort(docSimilarities, Collections.reverseOrder(new SimilarityComparator()));


        // Extract the sorted items in the doc_id and doc_sim lists
        for (int i = 0; i < docSimilarities.size(); i++) {
            docSimilarity ds = docSimilarities.get(i);
            if (ds.getSimilarity() <= upperLimit && low_index == -1) {
                low_index = i;
            }

            if (ds.getSimilarity() >= lowerLimit) {
                up_index = i;
            }
        }

        // Calculate the final score
        if(low_index == -1 || up_index == -1) {
            score = -1.0;
        } else{
            score = Double.parseDouble(low_index + "." + up_index);
        }        
        System.out.println(score);
    }
}
