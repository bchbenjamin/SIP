package com.sip.backend.domain;

import jakarta.persistence.*;
import java.time.ZonedDateTime;

@Entity
@Table(name = "incidents")
public class Incident {
    @Id
    @Column(name = "id")
    private String id;

    @Column(name = "state", nullable = false)
    private String state; // PENDING_VERIFICATION, VERIFIED, RESOLVED, etc.

    @Column(name = "threat_type", nullable = false)
    private String threatType;

    @Column(name = "threat_severity", nullable = false)
    private String threatSeverity;

    @Column(name = "threat_description")
    private String threatDescription;

    @Column(name = "latitude")
    private Double latitude;

    @Column(name = "longitude")
    private Double longitude;

    @Column(name = "location_readable")
    private String locationReadable;

    @Column(name = "location_accuracy")
    private Double locationAccuracy;

    @Column(name = "node_id", nullable = false)
    private String nodeId;

    @Column(name = "autopilot_handled", nullable = false)
    private boolean autopilotHandled;

    @Column(name = "created_at", nullable = false)
    private ZonedDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private ZonedDateTime updatedAt;

    // Getters and Setters...
}