package com.sip.guardian.data.local.entity;

import androidx.annotation.NonNull;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

/**
 * Room cache of the backend incidents subset (plan §17).
 * Complex nested objects are stored as JSON columns; media is cached via Coil,
 * not Room.
 */
@Entity(tableName = "incidents")
public class IncidentEntity {

    @PrimaryKey
    @NonNull
    public String id;
    public String state;
    public String threatType;
    public String threatSeverity;
    public String threatDescription;
    public double latitude;
    public double longitude;
    public String locationReadable;
    public String nodeId;
    public boolean autopilotHandled;
    public long createdAtEpochMs;
    public long updatedAtEpochMs;

    // JSON columns
    public String evidenceJson;
    public String detectionJson;
    public String annotationsJson;
    public String responseEventsJson;

    public boolean requiresVerification() {
        return "PENDING_VERIFICATION".equals(state);
    }
}
