package com.sip.guardian.domain.model;

import java.time.Instant;
import java.util.List;

/**
 * Core domain entity — a detected threat event.
 * OOP: Composition (built from Threat, Location, EvidenceBundle, DetectionResult...),
 * Encapsulation (state changes go through transitionTo, which enforces the machine).
 */
public final class Incident {
    private final String id;
    private final Threat threat;
    private final Location location;
    private final EvidenceBundle evidenceBundle;
    private final DetectionResult detectionResult;
    private final List<HumanAnnotation> humanAnnotations;
    private final List<ResponseEvent> responseEvents;
    private final String nodeId;
    private final Instant timestamp;
    private final boolean autopilotHandled;

    private IncidentState state;
    private Instant updatedAt;

    public Incident(String id, Threat threat, Location location, EvidenceBundle evidenceBundle,
                    DetectionResult detectionResult, List<HumanAnnotation> humanAnnotations,
                    List<ResponseEvent> responseEvents, IncidentState state, String nodeId,
                    Instant timestamp, boolean autopilotHandled, Instant updatedAt) {
        this.id = id;
        this.threat = threat;
        this.location = location;
        this.evidenceBundle = evidenceBundle;
        this.detectionResult = detectionResult;
        this.humanAnnotations = List.copyOf(humanAnnotations);
        this.responseEvents = List.copyOf(responseEvents);
        this.state = state;
        this.nodeId = nodeId;
        this.timestamp = timestamp;
        this.autopilotHandled = autopilotHandled;
        this.updatedAt = updatedAt;
    }

    public String getId() { return id; }
    public Threat getThreat() { return threat; }
    public Location getLocation() { return location; }
    public EvidenceBundle getEvidenceBundle() { return evidenceBundle; }
    public DetectionResult getDetectionResult() { return detectionResult; }
    public List<HumanAnnotation> getHumanAnnotations() { return humanAnnotations; }
    public List<ResponseEvent> getResponseEvents() { return responseEvents; }
    public IncidentState getState() { return state; }
    public String getNodeId() { return nodeId; }
    public Instant getTimestamp() { return timestamp; }
    public boolean isAutopilotHandled() { return autopilotHandled; }
    public Instant getUpdatedAt() { return updatedAt; }

    /**
     * Enforced state transition (plan §19). Throws on invalid transitions so the
     * UI can never drive the incident into an illegal state.
     */
    public void transitionTo(IncidentState next) {
        if (!state.canTransitionTo(next)) {
            throw new IllegalStateException(
                    "Illegal transition " + state + " -> " + next);
        }
        this.state = next;
        this.updatedAt = Instant.now();
    }

    /** True when a fresh operator verdict is still required. */
    public boolean requiresVerification() {
        return state == IncidentState.PENDING_VERIFICATION;
    }

    /** Latest human annotation, if any. */
    public HumanAnnotation latestAnnotation() {
        if (humanAnnotations.isEmpty()) return null;
        return humanAnnotations.get(humanAnnotations.size() - 1);
    }
}
