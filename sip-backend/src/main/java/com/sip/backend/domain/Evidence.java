package com.sip.backend.domain;

import jakarta.persistence.*;
import java.time.ZonedDateTime;

@Entity
@Table(name = "evidence")
public class Evidence {
    @Id
    @Column(name = "id")
    private String id;

    @Column(name = "incident_id", nullable = false)
    private String incidentId;

    @Column(name = "type", nullable = false)
    private String type; // image, video, audio, thumbnail

    @Column(name = "storage_key")
    private String storageKey;

    @Column(name = "mime_type")
    private String mimeType;

    @Column(name = "size_bytes")
    private Long sizeBytes;

    @Column(name = "sha256")
    private String sha256;

    @Column(name = "capture_timestamp", nullable = false)
    private ZonedDateTime captureTimestamp;

    @Column(name = "retention_expiry")
    private ZonedDateTime retentionExpiry;

    @Column(name = "upload_status", nullable = false)
    private String uploadStatus = "PENDING";

    @Column(name = "created_at", nullable = false)
    private ZonedDateTime createdAt;

    // Getters and Setters...
}