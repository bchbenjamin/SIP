package com.sip.backend.repository;

import com.sip.backend.domain.ResponseEvent;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface ResponseEventRepository extends JpaRepository<ResponseEvent, String> {
    List<ResponseEvent> findByIncidentId(String incidentId);
}