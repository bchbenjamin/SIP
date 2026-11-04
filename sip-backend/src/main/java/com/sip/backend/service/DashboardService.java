package com.sip.backend.service;

import com.sip.backend.domain.Incident;
import com.sip.backend.repository.IncidentRepository;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class DashboardService {
    private final IncidentRepository incidentRepository;
    // Should also inject NodeService

    public DashboardService(IncidentRepository incidentRepository) {
        this.incidentRepository = incidentRepository;
    }

    // Method to fetch data matching the contract (lines 661-674 in CONTEXT.md)
    public Map<String, Object> getDashboardData() {
        // TODO: Aggregate data: count nodes, count active threats, fetch recent incidents
        return Map.of(
                "totalNodes", 1,
                "onlineNodes", 1,
                "activeThreats", 0,
                "recentIncidents", List.of(),
                "systemHealth", "GREEN"
        );
    }
}