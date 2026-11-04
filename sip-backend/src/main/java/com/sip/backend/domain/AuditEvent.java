package com.sip.backend.domain;

import jakarta.persistence.*;
import java.time.ZonedDateTime;

@Entity
@Table(name = "audit_events")
public class AuditEvent {
    @Id
    @Column(name = "id")
    private String id;

    @Column(name = "incident_id")
    private String incidentId;

    @Column(name = "event_type", nullable = false)
    private String eventType; // e.g., INCIDENT_VERIFIED, POLICY_UPDATED

    @Column(name = "timestamp", nullable = false)
    private ZonedDateTime timestamp;

    @Column(name = "description")
    private String description;

    @Column(name = "actor")
    private String actor; // User ID or System Name

    // Getters and Setters...
}