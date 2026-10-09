package com.sip.backend.incident;

import com.sip.backend.entity.Incident;
import com.sip.backend.entity.HumanAnnotation;

import java.util.Map;
import java.util.Set;

public final class IncidentStateMachine {

    private static final Map<Incident.IncidentState, Set<Incident.IncidentState>> ALLOWED_TRANSITIONS =
            Map.ofEntries(
                    Map.entry(Incident.IncidentState.DETECTED, Set.of(
                            Incident.IncidentState.PENDING_VERIFICATION,
                            Incident.IncidentState.ESCALATED)),
                    Map.entry(Incident.IncidentState.PENDING_VERIFICATION, Set.of(
                            Incident.IncidentState.VERIFIED,
                            Incident.IncidentState.REJECTED,
                            Incident.IncidentState.ESCALATED)),
                    Map.entry(Incident.IncidentState.VERIFIED, Set.of(
                            Incident.IncidentState.ESCALATED)),
                    Map.entry(Incident.IncidentState.REJECTED, Set.of(
                            Incident.IncidentState.RESOLVED)),
                    Map.entry(Incident.IncidentState.ESCALATED, Set.of(
                            Incident.IncidentState.RESOLVED)),
                    Map.entry(Incident.IncidentState.RESOLVED, Set.of())
            );

    private IncidentStateMachine() {}

    public static boolean canTransition(Incident.IncidentState from, Incident.IncidentState to) {
        if (from == null || to == null || from == to) return false;
        Set<Incident.IncidentState> allowed = ALLOWED_TRANSITIONS.get(from);
        return allowed != null && allowed.contains(to);
    }

    public static void validateTransition(Incident.IncidentState from, Incident.IncidentState to) {
        if (!canTransition(from, to)) {
            throw new IllegalStateException("Invalid state transition from " + from + " to " + to);
        }
    }

    public static Incident.IncidentState nextStateForVerification(HumanAnnotation.HumanLabel label) {
        return switch (label) {
            case TRUE_POSITIVE -> Incident.IncidentState.VERIFIED;
            case FALSE_POSITIVE -> Incident.IncidentState.REJECTED;
            case UNCERTAIN -> Incident.IncidentState.ESCALATED;
        };
    }
}