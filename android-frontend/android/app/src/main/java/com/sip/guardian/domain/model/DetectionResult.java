package com.sip.guardian.domain.model;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * The original AI prediction for an incident.
 * OOP: Immutability — all fields final, NEVER overwritten after creation (plan §8).
 */
public final class DetectionResult {
    private final String predictedClass;
    private final double confidence;
    private final String modelVersion;
    private final Instant detectionTimestamp;
    private final List<String> sensorModalities;
    private final Map<String, Double> rawScores;

    public DetectionResult(String predictedClass, double confidence, String modelVersion,
                           Instant detectionTimestamp, List<String> sensorModalities,
                           Map<String, Double> rawScores) {
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("confidence must be 0..1");
        }
        this.predictedClass = predictedClass;
        this.confidence = confidence;
        this.modelVersion = modelVersion;
        this.detectionTimestamp = detectionTimestamp;
        this.sensorModalities = List.copyOf(sensorModalities);
        this.rawScores = Map.copyOf(rawScores);
    }

    public String getPredictedClass() { return predictedClass; }
    public double getConfidence() { return confidence; }
    public String getModelVersion() { return modelVersion; }
    public Instant getDetectionTimestamp() { return detectionTimestamp; }
    public List<String> getSensorModalities() { return sensorModalities; }
    public Map<String, Double> getRawScores() { return rawScores; }
}
