package com.ewb;

public class docSimilarity {
    private String id;
    private double similarity;

    public docSimilarity(String id, double similarity) {
        this.id = id;
        this.similarity = similarity;
    }

    public String getId() {
        return id;
    }

    public double getSimilarity() {
        return similarity;
    }
}