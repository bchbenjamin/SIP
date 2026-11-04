package com.sip.backend.controller;

import com.sip.backend.service.EvidenceService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/evidence")
public class EvidenceController {

    private final EvidenceService evidenceService;

    public EvidenceController(EvidenceService evidenceService) {
        this.evidenceService = evidenceService;
    }

    @GetMapping("/{incidentId}/image")
    public ResponseEntity<String> getImageEvidence(@PathVariable String incidentId) {
        // TODO: Authenticate, check incident state, fetch metadata, generate signed URL/stream
        return ResponseEntity.ok("Image evidence for " + incidentId);
    }

    @GetMapping("/{incidentId}/video")
    public ResponseEntity<String> getVideoEvidence(@PathVariable String incidentId) {
        // TODO: Authenticate, check incident state, fetch metadata, generate signed URL/stream
        return ResponseEntity.ok("Video evidence for " + incidentId);
    }

    // ... other endpoints (audio, thumbnail)
}