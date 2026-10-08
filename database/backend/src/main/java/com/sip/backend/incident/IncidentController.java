package com.sip.backend.incident;

import com.sip.backend.dto.*;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/v1/incidents")
public class IncidentController {

    private final IncidentService incidentService;

    public IncidentController(IncidentService incidentService) {
        this.incidentService = incidentService;
    }

    @GetMapping
    public ResponseEntity<PageDto<IncidentDto>> getIncidents(
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer size,
            @RequestParam(required = false) String state,
            @RequestParam(required = false) String threatType,
            @RequestParam(required = false) String nodeId,
            @RequestParam(required = false) String from,
            @RequestParam(required = false) String to) {
        return ResponseEntity.ok(incidentService.getIncidents(page, size, state, threatType, nodeId, from, to));
    }

    @GetMapping("/{id}")
    public ResponseEntity<IncidentDto> getIncident(@PathVariable String id) {
        return ResponseEntity.ok(incidentService.getIncident(id));
    }

    @PostMapping("/{id}/verify")
    @PreAuthorize("hasAnyRole('ADMIN', 'OPERATOR')")
    public ResponseEntity<IncidentDto> verify(
            @PathVariable String id,
            @Valid @RequestBody VerificationRequest request,
            Authentication auth) {
        return ResponseEntity.ok(incidentService.verify(id, request, auth.getName()));
    }

    @GetMapping("/{id}/events")
    public ResponseEntity<List<AuditEventDto>> getIncidentEvents(@PathVariable String id) {
        return ResponseEntity.ok(incidentService.getIncidentEvents(id));
    }
}