package com.sip.guardian.domain.model;

import java.time.Instant;
import java.util.Objects;

/**
 * Human operator's assessment. Stored separately from the AI prediction (plan §8).
 * Multiple annotations per incident are supported; each has a version.
 */
public final class HumanAnnotation {
    private final String id;
    private final HumanLabel label;
    private final String annotatorId;
    private final Instant timestamp;
    private final String notes;
    private final Double confidence;
    private final int version;

    public HumanAnnotation(String id, HumanLabel label, String annotatorId,
                           Instant timestamp, String notes, Double confidence, int version) {
        this.id = Objects.requireNonNull(id);
        this.label = Objects.requireNonNull(label);
        this.annotatorId = Objects.requireNonNull(annotatorId);
        this.timestamp = Objects.requireNonNull(timestamp);
        this.notes = notes == null ? "" : notes;
        this.confidence = confidence;
        this.version = version;
    }

    public String getId() { return id; }
    public HumanLabel getLabel() { return label; }
    public String getAnnotatorId() { return annotatorId; }
    public Instant getTimestamp() { return timestamp; }
    public String getNotes() { return notes; }
    public Double getConfidence() { return confidence; }
    public int getVersion() { return version; }

    /** Disagreement tracking (plan §8): AI vs human label mismatch. */
    public boolean disagreesWith(DetectionResult ai) {
        if (ai == null) return false;
        boolean aiPositive = !"NONE".equalsIgnoreCase(ai.getPredictedClass());
        return (label == HumanLabel.TRUE_POSITIVE) != aiPositive;
    }
}
