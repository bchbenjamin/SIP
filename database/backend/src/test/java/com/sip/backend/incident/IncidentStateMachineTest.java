package com.sip.backend.incident;

import com.sip.backend.entity.Incident;
import com.sip.backend.entity.HumanAnnotation;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class IncidentStateMachineTest {

    @Test
    void canTransition_DETECTED_to_PENDING_VERIFICATION() {
        assertTrue(IncidentStateMachine.canTransition(
            Incident.IncidentState.DETECTED, Incident.IncidentState.PENDING_VERIFICATION));
    }

    @Test
    void canTransition_DETECTED_to_ESCALATED() {
        assertTrue(IncidentStateMachine.canTransition(
            Incident.IncidentState.DETECTED, Incident.IncidentState.ESCALATED));
    }

    @Test
    void cannotTransition_DETECTED_to_VERIFIED() {
        assertFalse(IncidentStateMachine.canTransition(
            Incident.IncidentState.DETECTED, Incident.IncidentState.VERIFIED));
    }

    @Test
    void cannotTransition_RESOLVED_to_anything() {
        assertFalse(IncidentStateMachine.canTransition(
            Incident.IncidentState.RESOLVED, Incident.IncidentState.DETECTED));
    }

    @Test
    void cannotTransition_sameState() {
        assertFalse(IncidentStateMachine.canTransition(
            Incident.IncidentState.DETECTED, Incident.IncidentState.DETECTED));
    }

    @Test
    void validateTransition_throwsOnIllegalTransition() {
        assertThrows(IllegalStateException.class, () ->
            IncidentStateMachine.validateTransition(
                Incident.IncidentState.RESOLVED, Incident.IncidentState.DETECTED));
    }

    @Test
    void nextStateForVerification_TRUE_POSITIVE() {
        assertEquals(Incident.IncidentState.VERIFIED,
            IncidentStateMachine.nextStateForVerification(HumanAnnotation.HumanLabel.TRUE_POSITIVE));
    }

    @Test
    void nextStateForVerification_FALSE_POSITIVE() {
        assertEquals(Incident.IncidentState.REJECTED,
            IncidentStateMachine.nextStateForVerification(HumanAnnotation.HumanLabel.FALSE_POSITIVE));
    }

    @Test
    void nextStateForVerification_UNCERTAIN() {
        assertEquals(Incident.IncidentState.ESCALATED,
            IncidentStateMachine.nextStateForVerification(HumanAnnotation.HumanLabel.UNCERTAIN));
    }

    @Test
    void PENDING_VERIFICATION_canTransitionTo_VERIFIED_REJECTED_ESCALATED() {
        var from = Incident.IncidentState.PENDING_VERIFICATION;
        assertTrue(IncidentStateMachine.canTransition(from, Incident.IncidentState.VERIFIED));
        assertTrue(IncidentStateMachine.canTransition(from, Incident.IncidentState.REJECTED));
        assertTrue(IncidentStateMachine.canTransition(from, Incident.IncidentState.ESCALATED));
    }
    @Test
    void DETECTED_to_EVIDENCE_CAPTURED_to_AUTONOMOUS_EVALUATION_isAllowed() {
        assertTrue(IncidentStateMachine.canTransition(
            Incident.IncidentState.DETECTED, Incident.IncidentState.EVIDENCE_CAPTURED));
        assertTrue(IncidentStateMachine.canTransition(
            Incident.IncidentState.EVIDENCE_CAPTURED, Incident.IncidentState.AUTONOMOUS_EVALUATION));
        assertTrue(IncidentStateMachine.canTransition(
            Incident.IncidentState.AUTONOMOUS_EVALUATION, Incident.IncidentState.AUTO_HANDLED));
    }

    @Test
    void DETERRENCE_path_canCompleteOrEscalate() {
        assertTrue(IncidentStateMachine.canTransition(
            Incident.IncidentState.VERIFIED, Incident.IncidentState.DETERRENCE_ACTIVE));
        assertTrue(IncidentStateMachine.canTransition(
            Incident.IncidentState.DETERRENCE_ACTIVE, Incident.IncidentState.DETERRENCE_COMPLETED));
        assertTrue(IncidentStateMachine.canTransition(
            Incident.IncidentState.DETERRENCE_ACTIVE, Incident.IncidentState.ESCALATED));
        assertTrue(IncidentStateMachine.canTransition(
            Incident.IncidentState.DETERRENCE_COMPLETED, Incident.IncidentState.RESOLVED));
    }

}