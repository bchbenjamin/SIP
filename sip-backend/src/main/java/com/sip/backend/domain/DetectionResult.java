package com.sip.backend.domain;

import jakarta.persistence.*;
import java.time.ZonedDateTime;
import java.util.Map;

@Entity
@Table(name = "detection_results")
public class DetectionResult {
    @Id
    @Column(name = "id")
    private String id;

    @Column(name = "incident_id", nullable = false)
    private String incidentId;

    @Column(name = "predicted_class", nullable = false)
    private String predictedClass;

    @Column(name = "confidence", nullable = false)
    private Double confidence;

    @Column(name = "model_version", nullable = false)
    private String modelVersion;

    @Column(name = "detection_timestamp", nullable = false)
    private ZonedDateTime detectionTimestamp;

    @Column(name = "sensor_modalities")
    private String sensorModalities; // Storing as JSONB string for simplicity here

    @Column(name = "raw_scores")
    private String rawScores; // Storing as JSONB string for simplicity here

    // Getters and Setters...
}