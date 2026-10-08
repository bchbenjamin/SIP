package com.sip.guardian.domain.usecase;

import com.sip.guardian.domain.model.Incident;
import com.sip.guardian.domain.model.IncidentState;
import com.sip.guardian.domain.model.ThreatType;
import com.sip.guardian.domain.repository.IncidentRepository;

import java.time.Instant;
import java.util.List;

import javax.inject.Inject;

public class GetIncidentsUseCase {

    private final IncidentRepository incidentRepository;

    @Inject
    public GetIncidentsUseCase(IncidentRepository incidentRepository) {
        this.incidentRepository = incidentRepository;
    }

    public List<Incident> execute(Integer page, Integer size, IncidentState state,
                                  ThreatType threatType, String nodeId,
                                  Instant from, Instant to) {
        return incidentRepository.getIncidents(page, size, state, threatType, nodeId, from, to);
    }
}
