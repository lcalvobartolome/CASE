package com.ewb;
import java.util.Comparator;

public class SimilarityComparator implements Comparator<docSimilarity> {
    @Override
    public int compare(docSimilarity d1, docSimilarity d2) {
        if (d1.getSimilarity() < d2.getSimilarity()) {
            return -1;
        } else if (d1.getSimilarity() > d2.getSimilarity()) {
            return 1;
        }
        return 0;
    }
}