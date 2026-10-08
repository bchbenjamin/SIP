package com.sip.backend.controller;

import com.sip.backend.dto.PiIncidentPayload;
import com.sip.backend.entity.Incident;
import com.sip.backend.entity.Incident.IncidentState;
import com.sip.backend.repository.IncidentRepository;
import com.sip.backend.service.PiIncidentIngestionService;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/incidents")
public class IncidentController {

    private final IncidentRepository incidentRepository;
    private final PiIncidentIngestionService ingestionService;

    public IncidentController(IncidentRepository incidentRepository, PiIncidentIngestionService ingestionService) {
        this.incidentRepository = incidentRepository;
        this.ingestionService = ingestionService;
    }

    // ── Pi-facing REST ingestion (when MQTT is unavailable) ──────────────────

    @PostMapping
    public ResponseEntity<?> ingestIncident(@RequestBody PiIncidentPayload payload) {
        try {
            Incident incident = ingestionService.ingestIncident(payload);
            return ResponseEntity.status(201).body(incident);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }

    // ── Android/operator endpoints ───────────────────────────────────────────

    @GetMapping
    public ResponseEntity<Page<Incident>> listIncidents(
            @PageableDefault(size = 20) Pageable pageable) {
        return ResponseEntity.ok(incidentRepository.findAllByOrderByCreatedAtDesc(pageable));
    }

    @GetMapping("/{id}")
    public ResponseEntity<Incident> getIncident(@PathVariable String id) {
        return incidentRepository.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @PutMapping("/{id}/verify")
    public ResponseEntity<?> verify(@PathVariable String id) {
        return incidentRepository.findById(id).map(incident -> {
            incident.state = IncidentState.VERIFIED;
            incidentRepository.save(incident);
            return ResponseEntity.ok(incident);
        }).orElse(ResponseEntity.notFound().build());
    }

    @PutMapping("/{id}/reject")
    public ResponseEntity<?> reject(@PathVariable String id) {
        return incidentRepository.findById(id).map(incident -> {
            incident.state = IncidentState.REJECTED;
            incidentRepository.save(incident);
            return ResponseEntity.ok(incident);
        }).orElse(ResponseEntity.notFound().build());
    }

    @PutMapping("/{id}/escalate")
    public ResponseEntity<?> escalate(@PathVariable String id) {
        return incidentRepository.findById(id).map(incident -> {
            incident.state = IncidentState.ESCALATED;
            incidentRepository.save(incident);
            return ResponseEntity.ok(incident);
        }).orElse(ResponseEntity.notFound().build());
    }

    @PutMapping("/{id}/resolve")
    public ResponseEntity<?> resolve(@PathVariable String id) {
        return incidentRepository.findById(id).map(incident -> {
            incident.state = IncidentState.RESOLVED;
            incidentRepository.save(incident);
            return ResponseEntity.ok(incident);
        }).orElse(ResponseEntity.notFound().build());
    }
}