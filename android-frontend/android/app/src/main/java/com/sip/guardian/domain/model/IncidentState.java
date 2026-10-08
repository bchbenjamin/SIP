package com.sip.guardian.domain.model;

import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/**
 * Incident lifecycle states (plan §19).
 * Valid transitions are enforced here so the UI and use cases share one source of truth.
 */
public enum IncidentState {
    DETECTED,
    EVIDENCE_CAPTURED,
    PENDING_VERIFICATION,
    AUTONOMOUS_EVALUATION,
    AUTO_HANDLED,
    VERIFIED,
    REJECTED,
    DETERRENCE_ACTIVE,
    DETERRENCE_COMPLETED,
    ESCALATED,
    RESOLVED;

    private static final Map<IncidentState, Set<IncidentState>> TRANSITIONS;

    static {
        Map<IncidentState, Set<IncidentState>> m = new EnumMap<>(IncidentState.class);
        m.put(DETECTED, EnumSet.of(EVIDENCE_CAPTURED));
        m.put(EVIDENCE_CAPTURED, EnumSet.of(PENDING_VERIFICATION, AUTONOMOUS_EVALUATION));
        m.put(PENDING_VERIFICATION, EnumSet.of(VERIFIED, REJECTED));
        m.put(AUTONOMOUS_EVALUATION, EnumSet.of(AUTO_HANDLED, PENDING_VERIFICATION));
        m.put(VERIFIED, EnumSet.of(DETERRENCE_ACTIVE));
        m.put(AUTO_HANDLED, EnumSet.of(DETERRENCE_ACTIVE));
        m.put(DETERRENCE_ACTIVE, EnumSet.of(DETERRENCE_COMPLETED, ESCALATED));
        m.put(DETERRENCE_COMPLETED, EnumSet.of(RESOLVED));
        m.put(ESCALATED, EnumSet.of(RESOLVED));
        m.put(REJECTED, EnumSet.of(RESOLVED));
        m.put(RESOLVED, EnumSet.noneOf(IncidentState.class));
        TRANSITIONS = Collections.unmodifiableMap(m);
    }

    public boolean canTransitionTo(IncidentState next) {
        return TRANSITIONS.get(this).contains(next);
    }

    public Set<IncidentState> allowedTransitions() {
        return TRANSITIONS.get(this);
    }

    /** True when an operator may verify/reject from this state. */
    public boolean isVerifiable() {
        return this == PENDING_VERIFICATION;
    }

    /** True when deterrence output is expected/relevant for this state. */
    public boolean isDeterrencePath() {
        return this == VERIFIED || this == AUTO_HANDLED || this == DETERRENCE_ACTIVE
                || this == DETERRENCE_COMPLETED || this == ESCALATED;
    }
}
