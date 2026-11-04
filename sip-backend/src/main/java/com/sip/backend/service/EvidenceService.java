package com.sip.backend.service;

import com.sip.backend.repository.EvidenceRepository;
import com.sip.backend.domain.Evidence;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class EvidenceService {
    private final EvidenceRepository evidenceRepository;

    public EvidenceService(EvidenceRepository evidenceRepository) {
        this.evidenceRepository = evidenceRepository;
    }

    // Stubs for CRUD operations
    public void uploadEvidence(Evidence evidence) {
        // Logic: save metadata, return storage_key, check for duplicates
    }
    public List<Evidence> getEvidenceForIncident(String incidentId) {
        return evidenceRepository.findByIncidentId(incidentId);
    }
}