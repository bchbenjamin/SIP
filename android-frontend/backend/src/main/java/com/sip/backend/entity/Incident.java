package com.sip.backend.entity;

import jakarta.persistence.*;
import java.time.OffsetDateTime;

@Entity
@Table(name = "incidents")
public class Incident {
    @Id
    public String id;

    @Column(nullable = false)
    @Enumerated(EnumType.STRING)
    public IncidentState state = IncidentState.DETECTED;

    @Column(name = "threat_type", nullable = false)
    @Enumerated(EnumType.STRING)
    public ThreatType threatType;

    @Column(name = "threat_severity", nullable = false)
    @Enumerated(EnumType.STRING)
    public ThreatSeverity threatSeverity;

    @Column(name = "threat_description")
    public String threatDescription;

    public Double latitude;
    public Double longitude;

    @Column(name = "location_readable")
    public String locationReadable;

    @Column(name = "location_accuracy")
    public Double locationAccuracy;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "node_id", nullable = false)
    public Node node;

    @Column(name = "autopilot_handled", nullable = false)
    public Boolean autopilotHandled = false;

    @Column(name = "created_at", nullable = false)
    public OffsetDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    public OffsetDateTime updatedAt;

    public enum IncidentState {
        DETECTED, EVIDENCE_CAPTURED, PENDING_VERIFICATION, AUTONOMOUS_EVALUATION, AUTO_HANDLED, VERIFIED, REJECTED, DETERRENCE_ACTIVE, DETERRENCE_COMPLETED, ESCALATED, RESOLVED
    }

    public enum ThreatType {
        WEAPON, SUSPICIOUS_BEHAVIOR, AUDIO_ANOMALY, FIRE, UNKNOWN
    }

    public enum ThreatSeverity {
        DETERRABLE, NON_DETERRABLE
    }

    @PrePersist
    public void prePersist() {
        createdAt = OffsetDateTime.now();
        updatedAt = OffsetDateTime.now();
    }

    @PreUpdate
    public void preUpdate() {
        updatedAt = OffsetDateTime.now();
    }

    public Incident() {}
}