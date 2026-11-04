package com.sip.backend.domain;

import jakarta.persistence.*;
import java.time.ZonedDateTime;

@Entity
@Table(name = "nodes")
public class Node {
    @Id
    @Column(name = "id")
    private String id;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "status", nullable = false)
    private String status; // ONLINE, DEGRADED, OFFLINE, UNKNOWN

    @Column(name = "latitude")
    private Double latitude;

    @Column(name = "longitude")
    private Double longitude;

    @Column(name = "last_heartbeat")
    private ZonedDateTime lastHeartbeat;

    @Column(name = "battery_level")
    private Integer batteryLevel;

    @Column(name = "firmware_version")
    private String firmwareVersion;

    @Column(name = "created_at")
    private ZonedDateTime createdAt;

    @Column(name = "updated_at")
    private ZonedDateTime updatedAt;

    // Getters and Setters...
}