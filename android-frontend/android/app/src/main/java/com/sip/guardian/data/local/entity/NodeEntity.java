package com.sip.guardian.data.local.entity;

import androidx.annotation.NonNull;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "nodes")
public class NodeEntity {
    @PrimaryKey
    @NonNull
    public String nodeId;
    public String name;
    public String status;
    public double latitude;
    public double longitude;
    public long lastHeartbeatEpochMs;
    public Integer batteryLevel;
    public String firmwareVersion;
    public String policyJson;
}
