package com.sip.backend.domain;

import jakarta.persistence.*;
import java.time.ZonedDateTime;

@Entity
@Table(name = "autopilot_policies")
public class AutopilotPolicy {
    @Id
    @Column(name = "node_id", nullable = false)
    private String nodeId;

    @Column(name = "enabled", nullable = false)
    private boolean enabled = false;

    @Column(name = "allowed_threat_types")
    private String allowedThreatTypes; // JSONB string representation

    @Column(name = "confidence_threshold", nullable = false)
    private Double confidenceThreshold = 0.85;

    @Column(name = "deterrence_timeout_seconds", nullable = false)
    private Integer deterrenceTimeoutSeconds = 5;

    @Column(name = "auto_escalate_on_deterrence_failure", nullable = false)
    private boolean autoEscalateOnDeterrenceFailure = true;

    @Column(name = "max_deterrence_attempts", nullable = false)
    private Integer maxDeterrenceAttempts = 1;

    @Column(name = "always_escalate_types")
    private String alwaysEscalateTypes; // JSONB string representation

    @Column(name = "require_minimum_confidence", nullable = false)
    private boolean requireMinimumConfidence = true;

    @Column(name = "require_multi_modal_confirmation", nullable = false)
    private boolean requireMultiModalConfirmation = false;

    @Column(name = "never_autonomous_types")
    private String neverAutonomousTypes; // JSONB string representation

    @Column(name = "sync_state", nullable = false)
    private String syncState = "DISABLED";

    @Column(name = "last_sync")
    private ZonedDateTime lastSync;

    // Getters and Setters...
}