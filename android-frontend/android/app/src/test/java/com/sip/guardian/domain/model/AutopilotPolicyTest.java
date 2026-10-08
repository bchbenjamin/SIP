package com.sip.guardian.domain.model;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.Set;
import org.junit.Test;

public class AutopilotPolicyTest {
    @Test public void dangerousTypesAreBlockedByDefaultSafetyConstraints() {
        AutopilotPolicy policy = new AutopilotPolicy(
                true,
                Set.of(ThreatType.WEAPON, ThreatType.LOITERING),
                0.8,
                5,
                EscalationRules.defaults(),
                SafetyConstraints.defaults(),
                AutopilotState.ENABLED,
                null);
        assertFalse(policy.isAuthorizedFor(
                new Threat(ThreatType.WEAPON, ThreatSeverity.NON_DETERRABLE, "weapon"),
                0.99));
        assertTrue(policy.isAuthorizedFor(
                new Threat(ThreatType.LOITERING, ThreatSeverity.DETERRABLE, "loitering"),
                0.95));
    }
}
