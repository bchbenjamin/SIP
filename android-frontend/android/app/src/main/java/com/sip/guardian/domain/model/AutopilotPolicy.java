package com.sip.guardian.domain.model;

import java.time.Instant;
import java.util.Collections;
import java.util.Set;

/**
 * Autonomous operation policy for a node (plan §18).
 * The Pi stores the last valid policy locally; this class mirrors that model client-side.
 */
public final class AutopilotPolicy {
    private final boolean enabled;
    private final Set<ThreatType> allowedThreatTypes;
    private final double confidenceThreshold;          // 0.0 - 1.0
    private final int deterrenceTimeoutSeconds;
    private final EscalationRules escalationRules;
    private final SafetyConstraints safetyConstraints;
    private final AutopilotState syncState;
    private final Instant lastSyncTimestamp;

    public AutopilotPolicy(boolean enabled, Set<ThreatType> allowedThreatTypes,
                           double confidenceThreshold, int deterrenceTimeoutSeconds,
                           EscalationRules escalationRules, SafetyConstraints safetyConstraints,
                           AutopilotState syncState, Instant lastSyncTimestamp) {
        if (confidenceThreshold < 0.0 || confidenceThreshold > 1.0) {
            throw new IllegalArgumentException("confidenceThreshold must be 0..1");
        }
        this.enabled = enabled;
        this.allowedThreatTypes = Collections.unmodifiableSet(allowedThreatTypes);
        this.confidenceThreshold = confidenceThreshold;
        this.deterrenceTimeoutSeconds = deterrenceTimeoutSeconds;
        this.escalationRules = escalationRules;
        this.safetyConstraints = safetyConstraints;
        this.syncState = syncState;
        this.lastSyncTimestamp = lastSyncTimestamp;
    }

    public static AutopilotPolicy disabled() {
        return new AutopilotPolicy(false, Collections.emptySet(), 0.85, 5,
                EscalationRules.defaults(), SafetyConstraints.defaults(),
                AutopilotState.DISABLED, null);
    }

    public boolean isEnabled() { return enabled; }
    public Set<ThreatType> getAllowedThreatTypes() { return allowedThreatTypes; }
    public double getConfidenceThreshold() { return confidenceThreshold; }
    public int getDeterrenceTimeoutSeconds() { return deterrenceTimeoutSeconds; }
    public EscalationRules getEscalationRules() { return escalationRules; }
    public SafetyConstraints getSafetyConstraints() { return safetyConstraints; }
    public AutopilotState getSyncState() { return syncState; }
    public Instant getLastSyncTimestamp() { return lastSyncTimestamp; }

    /** Mirrors the Pi-side authorization check in edge_server autopilot evaluation. */
    public boolean isAuthorizedFor(Threat threat, double confidence) {
        return enabled
                && allowedThreatTypes.contains(threat.getType())
                && confidence >= confidenceThreshold
                && safetyConstraints.isSatisfied(threat);
    }

    public boolean meetsConfidenceThreshold(double confidence) {
        return confidence >= confidenceThreshold;
    }
}
