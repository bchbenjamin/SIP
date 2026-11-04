package com.sip.backend.domain;

import jakarta.persistence.*;
import java.time.ZonedDateTime;

@Entity
@Table(name = "human_annotations")
public class HumanAnnotation {
    @Id
    @Column(name = "id")
    private String id;

    @Column(name = "incident_id", nullable = false)
    private String incidentId;

    @Column(name = "annotator_id", nullable = false)
    private String annotatorId;

    @Column(name = "label", nullable = false)
    private String label;

    @Column(name = "timestamp", nullable = false)
    private ZonedDateTime timestamp;

    @Column(name = "notes")
    private String notes;

    @Column(name = "confidence")
    private Double confidence;

    @Column(name = "version", nullable = false)
    private Integer version = 1;

    // Getters and Setters...
}