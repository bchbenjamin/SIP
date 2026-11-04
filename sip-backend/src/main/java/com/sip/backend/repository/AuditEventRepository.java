package com.sip.backend.repository;

import com.sip.backend.domain.AuditEvent;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface AuditEventRepository extends JpaRepository<AuditEvent, String> {
    List<AuditEvent> findByIncidentId(String incidentId);
}