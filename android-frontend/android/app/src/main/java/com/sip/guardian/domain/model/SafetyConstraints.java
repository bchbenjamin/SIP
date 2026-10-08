package com.sip.guardian.domain.model;

import java.util.Collections;
import java.util.Set;

public final class SafetyConstraints {
    private final boolean requireMinimumConfidence;
    private final boolean requireMultiModalConfirmation;
    private final Set<ThreatType> neverAutonomousTypes;

    public SafetyConstraints(boolean requireMinimumConfidence,
                             boolean requireMultiModalConfirmation,
                             Set<ThreatType> neverAutonomousTypes) {
        this.requireMinimumConfidence = requireMinimumConfidence;
        this.requireMultiModalConfirmation = requireMultiModalConfirmation;
        this.neverAutonomousTypes = Collections.unmodifiableSet(neverAutonomousTypes);
    }

    public static SafetyConstraints defaults() {
        return new SafetyConstraints(true, true,
                Set.of(ThreatType.WEAPON, ThreatType.ASSAULT));
    }

    public boolean isRequireMinimumConfidence() { return requireMinimumConfidence; }
    public boolean isRequireMultiModalConfirmation() { return requireMultiModalConfirmation; }
    public Set<ThreatType> getNeverAutonomousTypes() { return neverAutonomousTypes; }

    /** WEAPON/ASSAULT etc. must always route to human verification (plan §18). */
    public boolean isSatisfied(Threat threat) {
        return !neverAutonomousTypes.contains(threat.getType());
    }
}
