package com.sip.backend.service;

import com.sip.backend.repository.AuditEventRepository;
import org.springframework.stereotype.Service;

@Service
public class AuditService {
    private final AuditEventRepository auditEventRepository;

    public AuditService(AuditEventRepository auditEventRepository) {
        this.auditEventRepository = auditEventRepository;
    }

    public void recordAudit(String incidentId, String eventType, String description, String actor) {
        // Logic: save AuditEvent, ensuring actor is authenticated user ID
    }
}