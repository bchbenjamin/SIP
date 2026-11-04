package com.sip.backend.repository;

import com.sip.backend.domain.Incident;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface IncidentRepository extends JpaRepository<Incident, String> {
    List<Incident> findByState(String state);
}