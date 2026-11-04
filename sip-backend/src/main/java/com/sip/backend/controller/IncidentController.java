package com.sip.backend.controller;

import com.sip.backend.auth.AuthController;
import com.sip.backend.service.IncidentService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1")
public class IncidentController {

    private final IncidentService incidentService;
    // Inject other services (e.g., AuditService, NodeService) here

    public IncidentController(IncidentService incidentService) {
        this.incidentService = incidentService;
    }

    @GetMapping("/incidents")
    public ResponseEntity<Map<String, Object>> getIncidents(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "50") int size) {
        // TODO: Implement pagination and return contract shape (lines 637-647 in CONTEXT.md)
        return ResponseEntity.ok(Map.of("content", List.of(), "page", page, "size", size, "totalElements", 0, "totalPages", 0));
    }

    @GetMapping("/incidents/{id}")
    public ResponseEntity<String> getIncidentDetails(@PathVariable String id) {
        // TODO: Implement lookup and error handling (e.g., INCIDENT_NOT_FOUND)
        return ResponseEntity.ok(incidentService.getIncidentDetails(id));
    }

    @PostMapping("/incidents/{id}/verify")
    public ResponseEntity<?> verifyIncident(@PathVariable String id) {
        // TODO: Trigger transactional verification logic via service
        // MUST validate: state=PENDING_VERIFICATION, correct user role, etc.
        incidentService.verifyIncident(id);
        return ResponseEntity.ok(Map.of("status", "Verification initiated"));
    }
}