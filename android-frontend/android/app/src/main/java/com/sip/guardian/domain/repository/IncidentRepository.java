package com.sip.guardian.domain.repository;

import com.sip.guardian.domain.model.HumanAnnotation;
import com.sip.guardian.domain.model.Incident;
import com.sip.guardian.domain.model.IncidentState;
import com.sip.guardian.domain.model.ThreatType;

import java.time.Instant;
import java.util.List;

import io.reactivex.rxjava3.core.Flowable;

/**
 * Contract for incident data access.
 * OOP: Dependency Inversion — domain depends on this interface, not on Retrofit/Room.
 */
public interface IncidentRepository {

    /** Filtered, paginated query (network-first, local fallback). */
    List<Incident> getIncidents(Integer page, Integer size, IncidentState state,
                                ThreatType threatType, String nodeId,
                                Instant from, Instant to);

    Incident getIncidentById(String id);

    /**
     * Records a human verdict and drives the state machine:
     * TRUE_POSITIVE -> VERIFIED, FALSE_POSITIVE/UNCERTAIN -> REJECTED.
     * The AI DetectionResult is never touched (plan §8).
     */
    Incident verifyIncident(String incidentId, HumanAnnotation annotation);

    /** Live observable of the incident list, fed by Room + WebSocket invalidations. */
    Flowable<List<Incident>> observeIncidents();

    Flowable<Incident> observeIncidentById(String id);
}
