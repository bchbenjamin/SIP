package com.sip.guardian.domain.model;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.Test;

public class IncidentStateTest {
    @Test public void validTransitionsAreAccepted() {
        assertTrue(IncidentState.DETECTED.canTransitionTo(IncidentState.EVIDENCE_CAPTURED));
        assertTrue(IncidentState.PENDING_VERIFICATION.canTransitionTo(IncidentState.VERIFIED));
        assertTrue(IncidentState.VERIFIED.canTransitionTo(IncidentState.DETERRENCE_ACTIVE));
    }

    @Test public void invalidTransitionsAreRejected() {
        assertFalse(IncidentState.DETECTED.canTransitionTo(IncidentState.VERIFIED));
        assertFalse(IncidentState.RESOLVED.canTransitionTo(IncidentState.DETECTED));
    }
}
