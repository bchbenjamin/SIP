package com.sip.guardian.domain.model;

import java.time.Instant;

/** A deterrence/escalation action taken in response to an incident. */
public final class ResponseEvent {
    private final String id;
    private final DeterrenceType actionType;
    private final Instant timestamp;
    private final String result;
    private final boolean autonomous;

    public ResponseEvent(String id, DeterrenceType actionType, Instant timestamp,
                         String result, boolean autonomous) {
        this.id = id;
        this.actionType = actionType;
        this.timestamp = timestamp;
        this.result = result == null ? "" : result;
        this.autonomous = autonomous;
    }

    public String getId() { return id; }
    public DeterrenceType getActionType() { return actionType; }
    public Instant getTimestamp() { return timestamp; }
    public String getResult() { return result; }
    public boolean isAutonomous() { return autonomous; }
}
