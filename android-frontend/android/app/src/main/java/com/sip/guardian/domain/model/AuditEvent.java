package com.sip.guardian.domain.model;

import java.time.Instant;

/** Immutable audit log entry — every sensitive action is logged with actor + timestamp. */
public final class AuditEvent {
    private final String id;
    private final String incidentId;
    private final String eventType;
    private final Instant timestamp;
    private final String description;
    private final String actor;

    public AuditEvent(String id, String incidentId, String eventType, Instant timestamp,
                      String description, String actor) {
        this.id = id;
        this.incidentId = incidentId;
        this.eventType = eventType;
        this.timestamp = timestamp;
        this.description = description;
        this.actor = actor;
    }

    public String getId() { return id; }
    public String getIncidentId() { return incidentId; }
    public String getEventType() { return eventType; }
    public Instant getTimestamp() { return timestamp; }
    public String getDescription() { return description; }
    public String getActor() { return actor; }
}
