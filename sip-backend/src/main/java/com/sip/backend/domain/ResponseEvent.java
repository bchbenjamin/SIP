package com.sip.backend.domain;

import jakarta.persistence.*;
import java.time.ZonedDateTime;

@Entity
@Table(name = "response_events")
public class ResponseEvent {
    @Id
    @Column(name = "id")
    private String id;

    @Column(name = "incident_id", nullable = false)
    private String incidentId;

    @Column(name = "action_type", nullable = false)
    private String actionType; // e.g., DETERRENCE_ACOUSTIC, ALERT_POLICE

    @Column(name = "timestamp", nullable = false)
    private ZonedDateTime timestamp;

    @Column(name = "result")
    private String result;

    @Column(name = "autonomous", nullable = false)
    private boolean autonomous = false;

    // Getters and Setters...
}