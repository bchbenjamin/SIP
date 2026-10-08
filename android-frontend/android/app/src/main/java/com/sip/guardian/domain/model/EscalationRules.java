package com.sip.guardian.domain.model;

import java.util.Collections;
import java.util.Set;

public final class EscalationRules {
    private final boolean autoEscalateOnDeterrenceFailure;
    private final int maxDeterrenceAttempts;
    private final Set<ThreatType> alwaysEscalateTypes;

    public EscalationRules(boolean autoEscalateOnDeterrenceFailure,
                           int maxDeterrenceAttempts, Set<ThreatType> alwaysEscalateTypes) {
        this.autoEscalateOnDeterrenceFailure = autoEscalateOnDeterrenceFailure;
        this.maxDeterrenceAttempts = maxDeterrenceAttempts;
        this.alwaysEscalateTypes = Collections.unmodifiableSet(alwaysEscalateTypes);
    }

    public static EscalationRules defaults() {
        return new EscalationRules(true, 3,
                Set.of(ThreatType.WEAPON, ThreatType.ASSAULT));
    }

    public boolean isAutoEscalateOnDeterrenceFailure() { return autoEscalateOnDeterrenceFailure; }
    public int getMaxDeterrenceAttempts() { return maxDeterrenceAttempts; }
    public Set<ThreatType> getAlwaysEscalateTypes() { return alwaysEscalateTypes; }
}
