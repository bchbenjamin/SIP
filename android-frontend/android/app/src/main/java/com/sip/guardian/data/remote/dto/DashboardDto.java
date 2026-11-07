package com.sip.guardian.data.remote.dto;

import com.google.gson.annotations.SerializedName;
import java.util.List;

public class DashboardDto {
    @SerializedName("totalNodes")
    public int totalNodes;

    @SerializedName("onlineNodes")
    public int onlineNodes;

    @SerializedName("activeThreats")
    public int activeThreats;

    @SerializedName("recentIncidents")
    public List<IncidentDto> recentIncidents;

    @SerializedName("systemHealth")
    public String systemHealth; // GREEN | YELLOW | RED
}
