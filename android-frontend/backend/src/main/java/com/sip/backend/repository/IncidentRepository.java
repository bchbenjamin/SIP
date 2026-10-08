package com.sip.backend.repository;

import com.sip.backend.entity.Incident;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import java.time.OffsetDateTime;
import java.util.List;

public interface IncidentRepository extends JpaRepository<Incident, String> {
    Page<Incident> findAllByOrderByCreatedAtDesc(Pageable pageable);

    Page<Incident> findByStateOrderByCreatedAtDesc(Incident.IncidentState state, Pageable pageable);

    Page<Incident> findByNodeIdOrderByCreatedAtDesc(String nodeId, Pageable pageable);

    Page<Incident> findByThreatTypeOrderByCreatedAtDesc(Incident.ThreatType threatType, Pageable pageable);

    @Query("""
        SELECT i FROM Incident i
        WHERE (:state IS NULL OR i.state = :state)
          AND (:nodeId IS NULL OR i.node.id = :nodeId)
          AND (:threatType IS NULL OR i.threatType = :threatType)
          AND (:from IS NULL OR i.createdAt >= :from)
          AND (:to IS NULL OR i.createdAt <= :to)
        ORDER BY i.createdAt DESC
        """)
    Page<Incident> search(
        @Param("state") Incident.IncidentState state,
        @Param("nodeId") String nodeId,
        @Param("threatType") Incident.ThreatType threatType,
        @Param("from") OffsetDateTime from,
        @Param("to") OffsetDateTime to,
        Pageable pageable
    );

    /** Limits dashboard results in SQL instead of loading the full incidents table. */
    List<Incident> findTop10ByOrderByCreatedAtDesc();

    long countByState(Incident.IncidentState state);

    @Query("SELECT i.state, COUNT(i) FROM Incident i GROUP BY i.state")
    List<Object[]> countByState();
}