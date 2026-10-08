package com.sip.guardian.data.remote.dto;

import java.util.List;

public class DashboardDto {
    public int totalNodes;
    public int onlineNodes;
    public int activeThreats;
    public List<IncidentDto> recentIncidents;
    public String systemHealth; // GREEN | YELLOW | RED
}
