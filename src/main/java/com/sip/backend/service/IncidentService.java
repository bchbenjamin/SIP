package com.sip.backend.service;

import com.sip.backend.domain.Incident;
import com.sip.backend.repository.IncidentRepository;
import com.sip.backend.repository.HumanAnnotationRepository;
import com.sip.backend.repository.AuditEventRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.Map;

@Service
public class IncidentService {
    private final IncidentRepository incidentRepository;
    private final AuditService auditService;
    private final HumanAnnotationRepository annotationRepository;
    // private final WebSocketBroadcaster broadcaster; // Injected later

    public IncidentService(IncidentRepository incidentRepository, AuditService auditService, HumanAnnotationRepository annotationRepository) {
        this.incidentRepository = incidentRepository;
        this.auditService = auditService;
        this.annotationRepository = annotationRepository;
    }

    @Transactional
    public void verifyIncident(String incidentId, String annotatorId, String label, Double confidence) {
        // 1. Fetch incident & validate state (PENDING_VERIFICATION)
        Incident incident = incidentRepository.findById(incidentId)
                .orElseThrow(() -> new RuntimeException("Incident not found: " + incidentId));

        if (!"PENDING_VERIFICATION".equals(incident.getState())) {
            throw new IllegalStateException("Cannot verify incident in state: " + incident.getState());
        }

        // 2. Validate user permissions (check role via security context) - STUBBED
        // if (!SecurityContextHolder.getContext().getAuthentication().getAuthorities().contains("OPERATOR")) { ... }

        // 3. Perform transactional update: state->VERIFIED, create HumanAnnotation, create AuditEvent
        incident.setState("VERIFIED");
        incidentRepository.save(incident);

        // Create Human Annotation
        // TODO: Map input to HumanAnnotation entity
        // annotationRepository.save(new HumanAnnotation(...));

        // Create Audit Event
        auditService.recordAudit(incidentId, "INCIDENT_VERIFIED", "Incident verified by " + annotatorId, annotatorId);

        // 4. Publish WebSocket event upon success
        // broadcaster.publishEvent(incidentId, "INCIDENT_VERIFIED");
    }

    public void updateIncidentState(String incidentId, String newState) {
        // Logic: state machine transition and audit logging
        // incidentRepository.findById(incidentId).ifPresent(i -> { i.setState(newState); incidentRepository.save(i); auditService.recordAudit(...); });
    }

    public String getIncidentDetails(String incidentId) {
        // Implementation using IncidentRepository
        return "Incident details for " + incidentId;
    }
}