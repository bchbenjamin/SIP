package com.sip.backend.repository;

import com.sip.backend.domain.HumanAnnotation;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface HumanAnnotationRepository extends JpaRepository<HumanAnnotation, String> {
    List<HumanAnnotation> findByIncidentId(String incidentId);
}