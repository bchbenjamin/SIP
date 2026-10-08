package com.sip.guardian.domain.model;

import java.time.Instant;

/** Metadata about a YOLO model version deployed to the Pis (dataset provenance). */
public final class ModelVersion {
    private final String version;
    private final String modelName;
    private final Instant trainedAt;
    private final String datasetHash;
    private final double validationMap50;

    public ModelVersion(String version, String modelName, Instant trainedAt,
                        String datasetHash, double validationMap50) {
        this.version = version;
        this.modelName = modelName;
        this.trainedAt = trainedAt;
        this.datasetHash = datasetHash;
        this.validationMap50 = validationMap50;
    }

    public String getVersion() { return version; }
    public String getModelName() { return modelName; }
    public Instant getTrainedAt() { return trainedAt; }
    public String getDatasetHash() { return datasetHash; }
    public double getValidationMap50() { return validationMap50; }
}
